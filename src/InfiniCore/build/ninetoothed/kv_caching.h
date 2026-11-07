#include "ninetoothed.h"
#ifdef __cplusplus
extern "C" NineToothedResult launch_kv_caching(NineToothedStream stream, NineToothedTensor k_cache, NineToothedTensor v_cache, NineToothedTensor k, NineToothedTensor v, NineToothedTensor past_lengths, int emb_dim_, int dtype_, int block_size_m_, int block_size_n_);
#else
NineToothedResult launch_kv_caching(NineToothedStream stream, NineToothedTensor k_cache, NineToothedTensor v_cache, NineToothedTensor k, NineToothedTensor v, NineToothedTensor past_lengths, int emb_dim_, int dtype_, int block_size_m_, int block_size_n_);
#endif
