#include "ninetoothed.h"

extern "C" NineToothedResult launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_16_16_contiguity_1_1_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_8, NineToothedTensor ninetoothed_tensor_9);
extern "C" NineToothedResult launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_16_16_contiguity_0_0_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_8, NineToothedTensor ninetoothed_tensor_9);
extern "C" NineToothedResult launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_1_1_contiguity_1_1_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_8, NineToothedTensor ninetoothed_tensor_9);
extern "C" NineToothedResult launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_1_1_contiguity_0_0_size_i32_stride_i32(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_8, NineToothedTensor ninetoothed_tensor_9);
extern "C" NineToothedResult launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_1_1_contiguity_0_0_size_i64_stride_i64(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_8, NineToothedTensor ninetoothed_tensor_9);

extern "C" NineToothedResult launch_relu_1_INFINI_DTYPE_F64_1024(NineToothedStream stream, NineToothedTensor ninetoothed_tensor_8, NineToothedTensor ninetoothed_tensor_9) {
    if (ninetoothed_tensor_8.shape[0] > 2147483647ULL || ninetoothed_tensor_8.strides[0] > 2147483647LL || ninetoothed_tensor_8.strides[0] < -2147483648LL || ninetoothed_tensor_9.shape[0] > 2147483647ULL || ninetoothed_tensor_9.strides[0] > 2147483647LL || ninetoothed_tensor_9.strides[0] < -2147483648LL) return launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_1_1_contiguity_0_0_size_i64_stride_i64(stream, ninetoothed_tensor_8, ninetoothed_tensor_9);
    if (ninetoothed_tensor_8.shape[0] % 16 == 0 && ninetoothed_tensor_9.shape[0] % 16 == 0 && ninetoothed_tensor_8.strides[0] == 1 && ninetoothed_tensor_9.strides[0] == 1) return launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_16_16_contiguity_1_1_size_i32_stride_i32(stream, ninetoothed_tensor_8, ninetoothed_tensor_9);
    if (ninetoothed_tensor_8.shape[0] % 16 == 0 && ninetoothed_tensor_9.shape[0] % 16 == 0) return launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_16_16_contiguity_0_0_size_i32_stride_i32(stream, ninetoothed_tensor_8, ninetoothed_tensor_9);
    if (ninetoothed_tensor_8.strides[0] == 1 && ninetoothed_tensor_9.strides[0] == 1) return launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_1_1_contiguity_1_1_size_i32_stride_i32(stream, ninetoothed_tensor_8, ninetoothed_tensor_9);
    return launch_relu_1_INFINI_DTYPE_F64_1024_divisibility_1_1_contiguity_0_0_size_i32_stride_i32(stream, ninetoothed_tensor_8, ninetoothed_tensor_9);
}
