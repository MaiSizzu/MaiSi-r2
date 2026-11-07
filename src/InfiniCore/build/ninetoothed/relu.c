#include "relu.h"

#include "infinicore.h"

#include "relu_1_INFINI_DTYPE_F16_1024.h"
#include "relu_1_INFINI_DTYPE_BF16_1024.h"
#include "relu_1_INFINI_DTYPE_F32_1024.h"
#include "relu_1_INFINI_DTYPE_F64_1024.h"
#include "relu_2_INFINI_DTYPE_F16_1024.h"
#include "relu_2_INFINI_DTYPE_BF16_1024.h"
#include "relu_2_INFINI_DTYPE_F32_1024.h"
#include "relu_2_INFINI_DTYPE_F64_1024.h"
#include "relu_3_INFINI_DTYPE_F16_1024.h"
#include "relu_3_INFINI_DTYPE_BF16_1024.h"
#include "relu_3_INFINI_DTYPE_F32_1024.h"
#include "relu_3_INFINI_DTYPE_F64_1024.h"
#include "relu_4_INFINI_DTYPE_F16_1024.h"
#include "relu_4_INFINI_DTYPE_BF16_1024.h"
#include "relu_4_INFINI_DTYPE_F32_1024.h"
#include "relu_4_INFINI_DTYPE_F64_1024.h"
#include "relu_5_INFINI_DTYPE_F16_1024.h"
#include "relu_5_INFINI_DTYPE_BF16_1024.h"
#include "relu_5_INFINI_DTYPE_F32_1024.h"
#include "relu_5_INFINI_DTYPE_F64_1024.h"

NineToothedResult launch_relu(NineToothedStream stream, NineToothedTensor input, NineToothedTensor output, int ndim_, int dtype_, int block_size_) {
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_relu_1_INFINI_DTYPE_F16_1024(stream, input, output);
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_relu_1_INFINI_DTYPE_BF16_1024(stream, input, output);
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_relu_1_INFINI_DTYPE_F32_1024(stream, input, output);
    if (ndim_ == 1 && dtype_ == INFINI_DTYPE_F64 && block_size_ == 1024)
        return launch_relu_1_INFINI_DTYPE_F64_1024(stream, input, output);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_relu_2_INFINI_DTYPE_F16_1024(stream, input, output);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_relu_2_INFINI_DTYPE_BF16_1024(stream, input, output);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_relu_2_INFINI_DTYPE_F32_1024(stream, input, output);
    if (ndim_ == 2 && dtype_ == INFINI_DTYPE_F64 && block_size_ == 1024)
        return launch_relu_2_INFINI_DTYPE_F64_1024(stream, input, output);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_relu_3_INFINI_DTYPE_F16_1024(stream, input, output);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_relu_3_INFINI_DTYPE_BF16_1024(stream, input, output);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_relu_3_INFINI_DTYPE_F32_1024(stream, input, output);
    if (ndim_ == 3 && dtype_ == INFINI_DTYPE_F64 && block_size_ == 1024)
        return launch_relu_3_INFINI_DTYPE_F64_1024(stream, input, output);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_relu_4_INFINI_DTYPE_F16_1024(stream, input, output);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_relu_4_INFINI_DTYPE_BF16_1024(stream, input, output);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_relu_4_INFINI_DTYPE_F32_1024(stream, input, output);
    if (ndim_ == 4 && dtype_ == INFINI_DTYPE_F64 && block_size_ == 1024)
        return launch_relu_4_INFINI_DTYPE_F64_1024(stream, input, output);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_F16 && block_size_ == 1024)
        return launch_relu_5_INFINI_DTYPE_F16_1024(stream, input, output);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_BF16 && block_size_ == 1024)
        return launch_relu_5_INFINI_DTYPE_BF16_1024(stream, input, output);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_F32 && block_size_ == 1024)
        return launch_relu_5_INFINI_DTYPE_F32_1024(stream, input, output);
    if (ndim_ == 5 && dtype_ == INFINI_DTYPE_F64 && block_size_ == 1024)
        return launch_relu_5_INFINI_DTYPE_F64_1024(stream, input, output);
    return INFINI_STATUS_NOT_IMPLEMENTED;
}
