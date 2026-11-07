#include "kv_caching.h"

#include "infinicore.h"

#include "kv_caching_1_INFINI_DTYPE_F16_64_64.h"
#include "kv_caching_1_INFINI_DTYPE_BF16_64_64.h"
#include "kv_caching_16_INFINI_DTYPE_BF16_64_64.h"
#include "kv_caching_16_INFINI_DTYPE_F32_64_64.h"

NineToothedResult launch_kv_caching(NineToothedStream stream, NineToothedTensor k_cache, NineToothedTensor v_cache, NineToothedTensor k, NineToothedTensor v, NineToothedTensor past_lengths, int emb_dim_, int dtype_, int block_size_m_, int block_size_n_) {
    if (emb_dim_ == 1 && dtype_ == INFINI_DTYPE_F16 && block_size_m_ == 64 && block_size_n_ == 64)
        return launch_kv_caching_1_INFINI_DTYPE_F16_64_64(stream, k_cache, v_cache, k, v, past_lengths);
    if (emb_dim_ == 1 && dtype_ == INFINI_DTYPE_BF16 && block_size_m_ == 64 && block_size_n_ == 64)
        return launch_kv_caching_1_INFINI_DTYPE_BF16_64_64(stream, k_cache, v_cache, k, v, past_lengths);
    if (emb_dim_ == 16 && dtype_ == INFINI_DTYPE_BF16 && block_size_m_ == 64 && block_size_n_ == 64)
        return launch_kv_caching_16_INFINI_DTYPE_BF16_64_64(stream, k_cache, v_cache, k, v, past_lengths);
    if (emb_dim_ == 16 && dtype_ == INFINI_DTYPE_F32 && block_size_m_ == 64 && block_size_n_ == 64)
        return launch_kv_caching_16_INFINI_DTYPE_F32_64_64(stream, k_cache, v_cache, k, v, past_lengths);
    return INFINI_STATUS_NOT_IMPLEMENTED;
}
