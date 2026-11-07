#include "ninetoothed.h"
#ifdef __cplusplus
extern "C" NineToothedResult launch_relu(NineToothedStream stream, NineToothedTensor input, NineToothedTensor output, int ndim_, int dtype_, int block_size_);
#else
NineToothedResult launch_relu(NineToothedStream stream, NineToothedTensor input, NineToothedTensor output, int ndim_, int dtype_, int block_size_);
#endif
