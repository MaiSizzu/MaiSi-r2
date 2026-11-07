#include "swiglu.h"

#include "infinicore.h"

#include "swiglu_1_INFINI_DTYPE_F16_1024.h"
#include "swiglu_1_INFINI_DTYPE_BF16_1024.h"
#include "swiglu_1_INFINI_DTYPE_F32_1024.h"
#include "swiglu_2_INFINI_DTYPE_F16_1024.h"
#include "swiglu_2_INFINI_DTYPE_BF16_1024.h"
#include "swiglu_2_INFINI_DTYPE_F32_1024.h"
#include "swiglu_3_INFINI_DTYPE_F16_1024.h"
#include "swiglu_3_INFINI_DTYPE_BF16_1024.h"
#include "swiglu_3_INFINI_DTYPE_F32_1024.h"
#include "swiglu_4_INFINI_DTYPE_F16_1024.h"
#include "swiglu_4_INFINI_DTYPE_BF16_1024.h"
#include "swiglu_4_INFINI_DTYPE_F32_1024.h"
#include "swiglu_5_INFINI_DTYPE_F16_1024.h"
#include "swiglu_5_INFINI_DTYPE_BF16_1024.h"
#include "swiglu_5_INFINI_DTYPE_F32_1024.h"

NineToothedResult launch_swiglu(NineToothedStream stream, NineToothedTensor output, NineToothedTensor up, NineToothedTensor gate, int ndim_, int dtype_, int block_size_) {
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_swiglu_1_INFINI_DTYPE_F16_1024(stream, output, up, gate);
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_swiglu_1_INFINI_DTYPE_BF16_1024(stream, output, up, gate);
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_swiglu_1_INFINI_DTYPE_F32_1024(stream, output, up, gate);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_swiglu_2_INFINI_DTYPE_F16_1024(stream, output, up, gate);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_swiglu_2_INFINI_DTYPE_BF16_1024(stream, output, up, gate);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_swiglu_2_INFINI_DTYPE_F32_1024(stream, output, up, gate);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_swiglu_3_INFINI_DTYPE_F16_1024(stream, output, up, gate);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_swiglu_3_INFINI_DTYPE_BF16_1024(stream, output, up, gate);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_swiglu_3_INFINI_DTYPE_F32_1024(stream, output, up, gate);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_swiglu_4_INFINI_DTYPE_F16_1024(stream, output, up, gate);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_swiglu_4_INFINI_DTYPE_BF16_1024(stream, output, up, gate);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_swiglu_4_INFINI_DTYPE_F32_1024(stream, output, up, gate);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_swiglu_5_INFINI_DTYPE_F16_1024(stream, output, up, gate);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_swiglu_5_INFINI_DTYPE_BF16_1024(stream, output, up, gate);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_swiglu_5_INFINI_DTYPE_F32_1024(stream, output, up, gate);
    return INFINI_STATUS_NOT_IMPLEMENTED;
}
