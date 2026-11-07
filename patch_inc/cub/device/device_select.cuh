

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SELECT_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SELECT_HPP_

#include "../config.hpp"

#include "../thread/thread_operators.cuh"

#include <cub/rocprim/device/device_select.hpp>

BEGIN_HIPCUB_NAMESPACE

class DeviceSelect
{
public:
    template <
        typename InputIteratorT,
        typename FlagIterator,
        typename OutputIteratorT,
        typename NumSelectedIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Flagged(void *d_temp_storage,
                       size_t &temp_storage_bytes,
                       InputIteratorT d_in,
                       FlagIterator d_flags,
                       OutputIteratorT d_out,
                       NumSelectedIteratorT d_num_selected_out,
                       int num_items,
                       cudaStream_t stream = 0,
                       bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::select(
            d_temp_storage, temp_storage_bytes,
            d_in, d_flags, d_out, d_num_selected_out, num_items,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT,
        typename SelectOp
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t If(void *d_temp_storage,
                  size_t &temp_storage_bytes,
                  InputIteratorT d_in,
                  OutputIteratorT d_out,
                  NumSelectedIteratorT d_num_selected_out,
                  int num_items,
                  SelectOp select_op,
                  cudaStream_t stream = 0,
                  bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::select(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, d_num_selected_out, num_items, select_op,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename NumSelectedIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Unique(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      NumSelectedIteratorT d_num_selected_out,
                      int num_items,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::unique(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, d_num_selected_out, num_items, cub::Equality(),
            stream, debug_synchronous
        );
    }

    template <
        typename KeyIteratorT,
        typename ValueIteratorT,
        typename OutputKeyIteratorT,
        typename OutputValueIteratorT,
        typename NumSelectedIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t UniqueByKey(void *d_temp_storage,
                           size_t &temp_storage_bytes,
                           KeyIteratorT d_keys_input,
                           ValueIteratorT d_values_input,
                           OutputKeyIteratorT d_keys_output,
                           OutputValueIteratorT d_values_output,
                           NumSelectedIteratorT d_num_selected_out,
                           int num_items,
                           cudaStream_t stream = 0,
                           bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::unique_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_input, d_values_input, 
            d_keys_output, d_values_output,
            d_num_selected_out, num_items, cub::Equality(),
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
