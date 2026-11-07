#include "flash_attention.h"

#include "infinicore.h"

#include "flash_attention_0_16_0_0_1_INFINI_DTYPE_F16_256_64.h"
#include "flash_attention_0_16_0_0_1_INFINI_DTYPE_BF16_256_64.h"

NineToothedResult launch_flash_attention(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_len, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int with_kv_cache_, int emb_dim_, int is_causal_, int with_attn_mask_, int causal_variant_, int dtype_, int block_size_m_, int block_size_n_) {
    if (with_kv_cache_ == 0 && emb_dim_ == 16 && is_causal_ == 0 && with_attn_mask_ == 0 && causal_variant_ == 1 && dtype_ == INFINI_DTYPE_F16 && block_size_m_ == 256 && block_size_n_ == 64)
        return launch_flash_attention_0_16_0_0_1_INFINI_DTYPE_F16_256_64(stream, query, key, value, total_kv_len, attn_mask, is_causal, output, with_attn_mask, causal_variant);
    if (with_kv_cache_ == 0 && emb_dim_ == 16 && is_causal_ == 0 && with_attn_mask_ == 0 && causal_variant_ == 1 && dtype_ == INFINI_DTYPE_BF16 && block_size_m_ == 256 && block_size_n_ == 64)
        return launch_flash_attention_0_16_0_0_1_INFINI_DTYPE_BF16_256_64(stream, query, key, value, total_kv_len, attn_mask, is_causal, output, with_attn_mask, causal_variant);
    return INFINI_STATUS_NOT_IMPLEMENTED;
}
