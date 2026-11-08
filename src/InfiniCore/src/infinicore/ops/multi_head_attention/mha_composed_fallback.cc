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

#if defined(ENABLE_HYGON_API) && !defined(ENABLE_FLASH_ATTN)

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
    Tensor probs = flat;
    if (p->is_causal) {
        op::causal_softmax_(flat, flat);
    } else {
        probs = op::softmax(flat, -1)
                    ->view({batch_size * num_kv_heads, ngroup * seq_len, seq_len});
    }

    auto attn_out = op::matmul(probs, V); // [B*H_kv, ngroup*S, D_v]

    auto merged = attn_out->view({batch_size, num_heads, seq_len, value_head_dim})
                      ->permute({0, 2, 1, 3})
                      ->contiguous()
                      ->view({batch_size, seq_len, num_heads * value_head_dim});

    out->copy_from(merged);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

namespace {

static bool registered = []() {
    // 这个编译分支下 hygon 没有任何 mha 注册，直接占位（override=true 只是为了让
    // 静态初始化顺序无关紧要——真实现存在时整个文件都不编译）。
    MultiheadAttention::plan_dispatcher().registerAll(&plan, true);
    MultiheadAttention::run_dispatcher().registerAll(&run, true);
    MultiheadAttention::cleanup_dispatcher().registerAll(&cleanup, true);
    return true;
}();

} // namespace

} // namespace infinicore::op::mha_impl::composed

#endif // ENABLE_HYGON_API && !ENABLE_FLASH_ATTN
