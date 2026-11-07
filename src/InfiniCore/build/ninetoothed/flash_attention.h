#include "ninetoothed.h"
#ifdef __cplusplus
extern "C" NineToothedResult launch_flash_attention(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_len, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int with_kv_cache_, int emb_dim_, int is_causal_, int with_attn_mask_, int causal_variant_, int dtype_, int block_size_m_, int block_size_n_);
#else
NineToothedResult launch_flash_attention(NineToothedStream stream, NineToothedTensor query, NineToothedTensor key, NineToothedTensor value, NineToothedTensor total_kv_len, NineToothedTensor attn_mask, NineToothedTensor is_causal, NineToothedTensor output, NineToothedTensor with_attn_mask, NineToothedTensor causal_variant, int with_kv_cache_, int emb_dim_, int is_causal_, int with_attn_mask_, int causal_variant_, int dtype_, int block_size_m_, int block_size_n_);
#endif
