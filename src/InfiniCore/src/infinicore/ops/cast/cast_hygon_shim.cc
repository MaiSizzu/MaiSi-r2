// Hygon(DCU) 上 infinicore::op::cast_ 的唯一实现走 ATen：
//   src/infinicore/ops/cast/cast.cc:38-47
//     void run_cast(...) {
//   #if defined(ENABLE_ATEN)
//       ... adaptor::to_aten_tensor(out).copy_(adaptor::to_aten_tensor(in));
//   #else
//       throw std::runtime_error("cast_ requires ATen");
//   #endif
//     }
// 本平台的构建没开 ENABLE_ATEN，所以只要有人调 cast_ 就必抛。
//
// DiT 第一次真的用到它是在 minimax_h3_transformer.cpp:525-526：
//   video_input = cast_to(video_hidden_states, F32)   // 入参本来就是 F32，短路
//   video_embeds = cast_to(linear_forward(proj_in_, video_input), dtype_)
// 入参被显式升到 F32，proj_in_ 的输出于是也是 F32，这里必须转回 bf16 —— 抛。
// 表现就是 8 个 rank 同时打 "exception during forward: cast_ requires ATen"，
// 上一层只看到 "RankWorker stopped during run"，一点线索都没有。
//
// 覆盖方式：独立 .so + LD_PRELOAD 同名符号（和 op::mha 兜底、cuda/ccl 薄壳同一条路）。
// _infinilm.so 对 cast_ 的调用是跨 DSO 的 PLT 调用，先加载者优先，
// 预加载的这份就顶掉了 cpp_api 里的那份。
//
// 为什么用 D2H -> CPU 转换 -> H2D，而不是写个 HIP kernel：
//   · DiT 里真正命中的转换只有 4 处中等张量（每步）：
//       proj_in 输出 2432×5376、audio_proj_in 输出 414×5376、norm_out 输出 2922×5376
//     外加每个 block 里 temb 那点（2×hidden，可忽略）；
//     一次 15M 元素的往返约 120MB，几十毫秒，相对 DiT 单步可忽略。
//   · kernel 要 hipcc + --offload-arch=gfx936+ 设备端自检，任何一个不对就是整单白跑；
//     这条路径只用到已经跑通的 infinirt memcpy（libh3cuda 薄壳早就把它接到 hip 上了）。
//
// 编译条件刻意只留 H3_CAST_STANDALONE：这份文件即使被 xmake 的 add_files 扫进
// cpp_api 的编译树，也是一个空 TU，绝不会和 cast.cc 撞符号。
#if defined(H3_CAST_STANDALONE)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/tensor.hpp"

namespace {

// bf16 只是 f32 的高 16 位，转换可以纯位运算做，不依赖任何半精度头文件。
// f32 -> bf16 用 round-to-nearest-even，和 ATen 的 BFloat16(float) 一致；
// 不做这一步（直接截断）会让 DiT 的数值和参考实现慢慢漂开。
inline float bf16_to_f32(uint16_t v) {
    const uint32_t u = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t exp = u & 0x7f800000u;
    if (exp == 0x7f800000u) { // inf / nan
        if ((u & 0x007fffffu) != 0) {
            return static_cast<uint16_t>((u >> 16) | 0x0040u); // 保留为 quiet NaN
        }
        return static_cast<uint16_t>(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

enum Kind { K_F32, K_BF16, K_OTHER };

Kind kind_of(const infinicore::DataType &d) {
    if (d == infinicore::DataType::F32) {
        return K_F32;
    }
    if (d == infinicore::DataType::BF16) {
        return K_BF16;
    }
    return K_OTHER;
}

// 只实现模型真会用到的那几种；其余明确报错，不要静默算出垃圾。
void convert_on_host(void *dst, const void *src, size_t n,
                     const infinicore::DataType &from,
                     const infinicore::DataType &to) {
    const Kind ks = kind_of(from);
    const Kind kd = kind_of(to);
    if (ks == K_F32 && kd == K_BF16) {
        const auto *s = static_cast<const float *>(src);
        auto *d = static_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) {
            d[i] = f32_to_bf16(s[i]);
        }
        return;
    }
    if (ks == K_BF16 && kd == K_F32) {
        const auto *s = static_cast<const uint16_t *>(src);
        auto *d = static_cast<float *>(dst);
        for (size_t i = 0; i < n; ++i) {
            d[i] = bf16_to_f32(s[i]);
        }
        return;
    }
    throw std::runtime_error("cast_ shim: 未实现的转换 " + infinicore::toString(from)
                             + " -> " + infinicore::toString(to)
                             + "（本 shim 只做 F32<->BF16）");
}

size_t g_calls = 0;

void log_call(size_t n, const infinicore::DataType &from,
              const infinicore::DataType &to, double mbytes) {
    const size_t k = ++g_calls;
    // 前几次全打（能看出调用顺序），之后每 500 次一行：
    // 日志里看到 [cast] 就证明符号确实被我们顶掉了；一条都没有 = 覆盖没生效。
    if (k <= 8 || k % 500 == 0) {
        std::fprintf(stderr, "[cast] #%zu %s->%s numel=%zu (%.1f MB)\n", k,
                     infinicore::toString(from).c_str(),
                     infinicore::toString(to).c_str(), n, mbytes);
    }
}

} // namespace

namespace infinicore::op {

// 与 include/infinicore/ops/cast.hpp 的声明逐字一致，靠同名符号顶掉 cpp_api 里那份。
void cast_(Tensor out, const Tensor &in) {
    if (!out || !in || out->shape() != in->shape()) {
        throw std::runtime_error("cast_ expects equal non-empty shapes");
    }
    if (out->device() != in->device()) {
        throw std::runtime_error("cast_: output 与 input 必须在同一设备上。out: "
                                 + out->info() + " in: " + in->info());
    }

    const auto &src_dtype = in->dtype();
    const auto &dst_dtype = out->dtype();

    // 转换要在调用方那个 rank 自己的设备上做（8 个 rank 是 8 个线程/进程，
    // hip 的 current device 是线程本地状态，必须自己设）。
    context::setDevice(out->device());

    // 非连续输入先整理成连续：后面的宿主转换按线性内存走。
    Tensor src = in;
    if (!src->is_contiguous()) {
        src = in->contiguous();
    }

    if (src_dtype == dst_dtype) {
        // 同类型就是纯拷贝，交给已有实现（它会处理 out 的 strides）。
        out->copy_from(src);
        return;
    }

    const size_t n = src->numel();
    if (n == 0) {
        return;
    }
    if (static_cast<size_t>(out->numel()) != n) {
        throw std::runtime_error("cast_: numel 不一致。out: " + out->info()
                                 + " in: " + in->info());
    }

    const size_t s_size = src->element_size();
    const size_t d_size = out->element_size();
    log_call(n, src_dtype, dst_dtype,
             static_cast<double>(n) * static_cast<double>(s_size + d_size) / (1024.0 * 1024.0));

    std::vector<uint8_t> host_in(n * s_size);
    std::vector<uint8_t> host_out(n * d_size);

    if (out->device().getType() == Device::Type::CPU) {
        convert_on_host(host_out.data(), src->data(), n, src_dtype, dst_dtype);
        std::memcpy(out->data(), host_out.data(), n * d_size);
        return;
    }

    // 关键：先把生产者的队列排空。infinirt 的 D2H 走的是 cudaMemcpy(同步)，
    // 它和 infinirt 自己那条 stream 之间没有顺序保证，不 sync 可能读到旧值。
    context::syncStream();
    context::memcpyD2H(host_in.data(), src->data(), n * s_size);
    convert_on_host(host_out.data(), host_in.data(), n, src_dtype, dst_dtype);

    // H2D 用阻塞版：host_out 出了这个函数就析构，异步拷贝不安全。
    if (out->is_contiguous()) {
        context::memcpyH2D(out->data(), host_out.data(), n * d_size, /*async=*/false);
    } else {
        Tensor tmp = Tensor::empty(out->shape(), dst_dtype, out->device());
        context::memcpyH2D(tmp->data(), host_out.data(), n * d_size, /*async=*/false);
        out->copy_from(tmp);
    }
}

} // namespace infinicore::op

namespace {
// 预加载自检用：构造期打一行，确认这份 .so 真的被 ld.so 挂上了。
// （符号有没有被顶掉，要看运行时的 [cast] 行。）
__attribute__((constructor, visibility("hidden"))) void h3_cast_shim_banner() {
    std::fprintf(stderr,
                 "[cast] cast_ 覆盖已就位（Hygon 无 ATen，F32<->BF16 走 D2H->CPU->H2D）\n");
}
} // namespace

#endif // H3_CAST_STANDALONE
