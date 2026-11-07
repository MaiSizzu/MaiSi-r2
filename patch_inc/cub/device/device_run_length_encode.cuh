

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include "../config.hpp"

#include <cub/rocprim/device/device_run_length_encode.hpp>

BEGIN_HIPCUB_NAMESPACE

class DeviceRunLengthEncode
{
public:
    template<
        typename InputIteratorT,
        typename UniqueOutputIteratorT,
        typename LengthsOutputIteratorT,
        typename NumRunsOutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Encode(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      UniqueOutputIteratorT d_unique_out,
                      LengthsOutputIteratorT d_counts_out,
                      NumRunsOutputIteratorT d_num_runs_out,
                      int num_items,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::run_length_encode(
            d_temp_storage, temp_storage_bytes,
            d_in, num_items,
            d_unique_out, d_counts_out, d_num_runs_out,
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OffsetsOutputIteratorT,
        typename LengthsOutputIteratorT,
        typename NumRunsOutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t NonTrivialRuns(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              InputIteratorT d_in,
                              OffsetsOutputIteratorT d_offsets_out,
                              LengthsOutputIteratorT d_lengths_out,
                              NumRunsOutputIteratorT d_num_runs_out,
                              int num_items,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::run_length_encode_non_trivial_runs(
            d_temp_storage, temp_storage_bytes,
            d_in, num_items,
            d_offsets_out, d_lengths_out, d_num_runs_out,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
