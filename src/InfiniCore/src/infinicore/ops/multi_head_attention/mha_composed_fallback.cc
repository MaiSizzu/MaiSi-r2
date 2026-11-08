// Hygon(DCU) 上没有 flash-attn 扩展时的 MultiheadAttention 兜底实现。
//
// 背景：xmake.lua 只有在 --flash-attn 不为空时才保留 ops/*/hygon/*.cc，
// 而 hygon/mha_flashattn_hygon.cc 又被 `#if defined(ENABLE_HYGON_API) &&
// defined(ENABLE_FLASH_ATTN)` 包住；通用版 mha_flashattn.cc 只编 NVIDIA/METAX/QY。
// 于是 README 的标准构建（--hygon-dcu=y --ninetoothed=y --ccl=y，无 --flash-attn）
// 下 MultiheadAttention 的 dispatcher 里 **一个函数指针都没有**，
// graph.hpp 的 INFINICORE_GRAPH_OP_DISPATCH 又不判空，直接 call 0 -> SIGSEGV(rip=0)。
//
// 这里用已注册的基础算子把注意力算出来（和 InfiniLM 的 static_attn 后端同一条数学路径）：
//     q·k^T (matmul, alpha=scale) -> softmax/causal_softmax -> ·v (matmul)
// 输入输出布局沿用调用方约定：[B, S, H, D]（vision 塔与 DiT 都是这么传的）。
//
// 编译条件刻意写成「Hygon 且没有 ENABLE_FLASH_ATTN」：只有这一种配置是坏的。
// 一旦有人用 --flash-attn 编（走 hygon/mha_flashattn_hygon.cc 的真实现），
// 本文件整体不参与编译，绝不会顶掉快路径。

// 两种编译形态，代码完全相同：
//   · 编进 libinfinicore_cpp_api.so（宏 ENABLE_HYGON_API）—— 需要重编该库时才走这条；
//   · 独立编成 libh3mha.so，用 LD_PRELOAD 挂（宏 H3_MHA_STANDALONE）—— 只编一个 .cc，
//     靠构造期把实现注册进 cpp_api 里的 dispatcher。后者是当前实际使用的路径：
//     重编 cpp_api 会连带触发 infiniop 全量重编（约 38 分钟），单作业 50 分钟装不下。
#if (defined(ENABLE_HYGON_API) || defined(H3_MHA_STANDALONE)) && !defined(ENABLE_FLASH_ATTN)

#include <cstdio>

#include "infinicore/ops/mha.hpp"

#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/softmax.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace infinicore::op::mha_impl::composed {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    // 调用点（qwen3_5_vision / minimax_h3_transformer）都是 4D + 无 alibi；
    // 真出现别的用法时宁可在 plan 阶段报错，也不要算出错误结果。
    if (q->shape().size() != 4) {
        throw std::runtime_error(
            "MultiheadAttention(composed fallback): expect 4D [B,S,H,D] tensors, got rank "
            + std::to_string(q->shape().size()));
    }
    if (alibi_slopes.has_value()) {
        throw std::runtime_error(
            "MultiheadAttention(composed fallback): alibi_slopes is not supported");
    }

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        scale,
        is_causal};
}

namespace {

std::string shape_str(const Tensor &t) {
    std::string s = "[";
    for (size_t i = 0; i < t->shape().size(); ++i) {
        if (i) s += ",";
        s += std::to_string(t->shape()[i]);
    }
    return s + "]";
}

void note_once() {
    static bool done = false;
    if (!done) {
        done = true;
        std::fprintf(stderr,
                     "[mha] 使用 InfiniCore 组合实现（matmul+softmax+matmul），"
                     "本平台无 flash-attn 扩展\n");
    }
}

} // namespace

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    Tensor out(p->out);
    Tensor q(p->q);
    Tensor k(p->k);
    Tensor v(p->v);

    const auto &qs = q->shape();
    const size_t batch_size = qs[0];
    const size_t seq_len = qs[1];
    const size_t num_heads = qs[2];
    const size_t head_dim = qs[3];

    const size_t num_kv_heads = k->size(2);
    const size_t value_head_dim = v->size(3);

    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error(
            "MultiheadAttention(composed fallback): num_heads " + std::to_string(num_heads)
            + " is not a multiple of num_kv_heads " + std::to_string(num_kv_heads)
            + " q=" + shape_str(q) + " k=" + shape_str(k) + " v=" + shape_str(v));
    }

    note_once();

    const size_t ngroup = num_heads / num_kv_heads;

    // [B, S, H, D] -> [B, H, S, D] -> [B*H_kv, ngroup*S, D]
    auto Q = q->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_kv_heads, ngroup * seq_len, head_dim});
    auto K = k->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_kv_heads, seq_len, head_dim});
    auto V = v->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_kv_heads, seq_len, value_head_dim});

    auto K_transposed = K->permute({0, 2, 1}); // [B*H_kv, D, S]

    // alpha 直接折进 matmul，省一次全量缩放
    auto attn_weight = op::matmul(Q, K_transposed, p->scale); // [B*H_kv, ngroup*S, S]

    auto flat = attn_weight->view({batch_size * num_heads, seq_len, seq_len});
    // causal 用 causal_softmax_（static_attn 已验证过原地可用）；
    // 非 causal 走独立的 softmax，避免 out/in 同一块内存时还要赌实现里的原地语义。
    //
    // 之后必须 view 回 GQA 展开的形状再乘 V：probs 的 batch 维得是 B*H_kv，
    // 留成 [B*H_q, S, S] 去乘 [B*H_kv, S, D_v]，ngroup>1 时 batch 维直接对不上。
    Tensor probs = flat->view({batch_size * num_kv_heads, ngroup * seq_len, seq_len});
    if (p->is_causal) {
        op::causal_softmax_(flat, flat);
    } else {
        probs = op::softmax(flat, -1)
                    ->view({batch_size * num_kv_heads, ngroup * seq_len, seq_len});
    }

    auto attn_out = op::matmul(probs, V); // [B*H_kv, ngroup*S, D_v]

    auto merged = attn_out->view({batch_size, num_heads, seq_len, value_head_dim})
                      ->permute({0, 2, 1, 3})
                      ->contiguous(); // [B, S, H, D_v]：必须和 out 同形

    // copy_from 的实现是 `if (src->shape() != this->shape()) throw`（copy.cc:21），
    // 所以这里千万不要 view 成 [B,S,H*D] —— 元素数一样也不行，形状必须逐维相同。
    out->copy_from(merged);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

namespace {

void register_all() {
    // override=false：只填没人注册的槽位。Hygon 上这个表本来就是全 nullptr，
    // 所以一定会填上；将来若真编了 --flash-attn 的真实现，也不会被我们顶掉。
    MultiheadAttention::plan_dispatcher().registerAll(&plan, false);
    MultiheadAttention::run_dispatcher().registerAll(&run, false);
    MultiheadAttention::cleanup_dispatcher().registerAll(&cleanup, false);
    // 这行是判断「兜底到底有没有装上」的唯一凭据，日志里搜 [mha] 即可。
    std::fprintf(stderr, "[mha] 兜底已注册到 dispatcher（Hygon，无 flash-attn）\n");
}

} // namespace

#if defined(H3_MHA_STANDALONE)
// 独立 .so 走 LD_PRELOAD：构造期注册到 libinfinicore_cpp_api.so 里的 dispatcher。
// 链接依赖保证 cpp_api 先加载，所以此处调用 plan_dispatcher() 是安全的。
// try/catch 兜住一切异常：这个构造函数跑在所有 python 进程里（包括预检），
// 绝不能因为注册失败把进程带崩。
// 注意：这里**不能**写 static —— C++ 标准禁止在 extern "C" 的链接规范里带存储类，
// gcc 报 `invalid use of 'static' in linkage specification`（14:49 那单就死在这一个词）。
// 用 visibility("hidden") 把它藏起来，效果和 static 一样，但不违反链接规范。
extern "C" __attribute__((constructor, visibility("hidden"))) void h3_mha_fallback_ctor() {
    try {
        register_all();
        // 注册完立刻回查：静态 dispatcher 是函数内 static，正常只会有一份；
        // 万一拿到的不是 cpp_api 那份（跨 DSO 各一份），注册就是白做。
        // 这一行是「兜底到底有没有真装上」的唯一凭据，日志里搜「注册回查」即可。
        auto pfn = MultiheadAttention::plan_dispatcher().lookup(Device::Type::HYGON);
        auto rfn = MultiheadAttention::run_dispatcher().lookup(Device::Type::HYGON);
        std::fprintf(stderr,
                     "[mha] 注册回查：plan=%s run=%s\n",
                     pfn ? "非空" : "空(注册没落到 cpp_api 那份 dispatcher)",
                     rfn ? "非空" : "空(注册没落到 cpp_api 那份 dispatcher)");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[mha] 注册失败：%s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[mha] 注册失败（未知异常），保持不注册状态\n");
    }
}
#else
namespace {
const bool registered = (register_all(), true);
} // namespace
#endif

} // namespace infinicore::op::mha_impl::composed

#endif // (ENABLE_HYGON_API || H3_MHA_STANDALONE) && !ENABLE_FLASH_ATTN
