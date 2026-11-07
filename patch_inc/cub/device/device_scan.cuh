

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SCAN_HPP_

#include <iostream>
#include "../config.hpp"

#include "../thread/thread_operators.cuh"

#include <cub/rocprim/device/device_scan.hpp>
#include <cub/rocprim/device/device_scan_by_key.hpp>

BEGIN_HIPCUB_NAMESPACE

class DeviceScan
{
public:
    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t InclusiveSum(void *d_temp_storage,
                            size_t &temp_storage_bytes,
                            InputIteratorT d_in,
                            OutputIteratorT d_out,
                            size_t num_items,
                            cudaStream_t stream = 0,
                            bool debug_synchronous = false)
    {
        return InclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, cub::Sum(), num_items,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t InclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             size_t num_items,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::inclusive_scan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items,
            scan_op,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ExclusiveSum(void *d_temp_storage,
                            size_t &temp_storage_bytes,
                            InputIteratorT d_in,
                            OutputIteratorT d_out,
                            size_t num_items,
                            cudaStream_t stream = 0,
                            bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return ExclusiveScan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, cub::Sum(), T(0), num_items,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT,
        typename InitValueT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ExclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             InitValueT init_value,
                             size_t num_items,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::exclusive_scan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init_value, num_items,
            scan_op,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ScanOpT,
        typename InitValueT,
        typename InitValueIterT = InitValueT*
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ExclusiveScan(void *d_temp_storage,
                             size_t &temp_storage_bytes,
                             InputIteratorT d_in,
                             OutputIteratorT d_out,
                             ScanOpT scan_op,
                             FutureValue<InitValueT, InitValueIterT> init_value,
                             int num_items,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::exclusive_scan(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init_value, num_items,
            scan_op,
            stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename EqualityOpT = cub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ExclusiveSumByKey(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 KeysInputIteratorT d_keys_in,
                                 ValuesInputIteratorT d_values_in,
                                 ValuesOutputIteratorT d_values_out,
                                 int num_items,
                                 EqualityOpT equality_op = EqualityOpT(),
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false)
    {
        using in_value_type = typename std::iterator_traits<ValuesInputIteratorT>::value_type;

        return (cudaError_t)::rocprim::exclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            static_cast<in_value_type>(0), static_cast<size_t>(num_items),
            cub::Sum(), equality_op, stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename ScanOpT,
        typename InitValueT,
        typename EqualityOpT = cub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ExclusiveScanByKey(void *d_temp_storage,
                                  size_t &temp_storage_bytes,
                                  KeysInputIteratorT d_keys_in,
                                  ValuesInputIteratorT d_values_in,
                                  ValuesOutputIteratorT d_values_out,
                                  ScanOpT scan_op,
                                  InitValueT init_value,
                                  int num_items,
                                  EqualityOpT equality_op = EqualityOpT(),
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::exclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            init_value, static_cast<size_t>(num_items),
            scan_op, equality_op, stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename EqualityOpT = cub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t InclusiveSumByKey(void *d_temp_storage,
                                 size_t &temp_storage_bytes,
                                 KeysInputIteratorT d_keys_in,
                                 ValuesInputIteratorT d_values_in,
                                 ValuesOutputIteratorT d_values_out,
                                 int num_items,
                                 EqualityOpT equality_op = EqualityOpT(),
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::inclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            static_cast<size_t>(num_items), cub::Sum(),
            equality_op, stream, debug_synchronous
        );
    }

    template <
        typename KeysInputIteratorT,
        typename ValuesInputIteratorT,
        typename ValuesOutputIteratorT,
        typename ScanOpT,
        typename EqualityOpT = cub::Equality
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t InclusiveScanByKey(void *d_temp_storage,
                                  size_t &temp_storage_bytes,
                                  KeysInputIteratorT d_keys_in,
                                  ValuesInputIteratorT d_values_in,
                                  ValuesOutputIteratorT d_values_out,
                                  ScanOpT scan_op,
                                  int num_items,
                                  EqualityOpT equality_op = EqualityOpT(),
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::inclusive_scan_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, d_values_out,
            static_cast<size_t>(num_items), scan_op,
            equality_op, stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
