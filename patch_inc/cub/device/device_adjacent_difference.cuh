

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_HPP_

#include "../config.hpp"

#include <cub/thread/thread_operators.cuh>
#include <cub/rocprim/device/device_adjacent_difference.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DeviceAdjacentDifference
{
    template <typename InputIteratorT,
              typename OutputIteratorT,
              typename DifferenceOpT = cub::Difference>
    static HIPCUB_RUNTIME_FUNCTION cudaError_t
    SubtractLeftCopy(void *d_temp_storage,
                     std::size_t &temp_storage_bytes,
                     InputIteratorT d_input,
                     OutputIteratorT d_output,
                     std::size_t num_items,
                     DifferenceOpT difference_op = {},
                     cudaStream_t stream = 0,
                     bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::adjacent_difference(
            d_temp_storage, temp_storage_bytes, d_input, d_output,
            num_items, difference_op, stream, debug_synchronous
        );
    }

    template <typename RandomAccessIteratorT,
              typename DifferenceOpT = cub::Difference>
    static HIPCUB_RUNTIME_FUNCTION cudaError_t
    SubtractLeft(void *d_temp_storage,
                 std::size_t &temp_storage_bytes,
                 RandomAccessIteratorT d_input,
                 std::size_t num_items,
                 DifferenceOpT difference_op = {},
                 cudaStream_t stream = 0,
                 bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::adjacent_difference_inplace(
            d_temp_storage, temp_storage_bytes, d_input,
            num_items, difference_op, stream, debug_synchronous
        );
    }

    template <typename InputIteratorT,
              typename OutputIteratorT,
              typename DifferenceOpT = cub::Difference>
    static HIPCUB_RUNTIME_FUNCTION cudaError_t
    SubtractRightCopy(void *d_temp_storage,
                      std::size_t &temp_storage_bytes,
                      InputIteratorT d_input,
                      OutputIteratorT d_output,
                      std::size_t num_items,
                      DifferenceOpT difference_op = {},
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::adjacent_difference_right(
            d_temp_storage, temp_storage_bytes, d_input, d_output,
            num_items, difference_op, stream, debug_synchronous
        );
    }

    template <typename RandomAccessIteratorT,
              typename DifferenceOpT = cub::Difference>
    static HIPCUB_RUNTIME_FUNCTION cudaError_t
    SubtractRight(void *d_temp_storage,
                  std::size_t &temp_storage_bytes,
                  RandomAccessIteratorT d_input,
                  std::size_t num_items,
                  DifferenceOpT difference_op = {},
                  cudaStream_t stream = 0,
                  bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::adjacent_difference_right_inplace(
            d_temp_storage, temp_storage_bytes, d_input,
            num_items, difference_op, stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
