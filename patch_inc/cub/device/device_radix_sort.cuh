

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_

#include "../config.hpp"

#include "../util_type.cuh"

#include <cub/rocprim/device/device_radix_sort.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DeviceRadixSort
{
    template<typename KeyT, typename ValueT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairs(void * d_temp_storage,
                         size_t& temp_storage_bytes,
                         const KeyT * d_keys_in,
                         KeyT * d_keys_out,
                         const ValueT * d_values_in,
                         ValueT * d_values_out,
                         NumItemsT num_items,
                         int begin_bit = 0,
                         int end_bit = sizeof(KeyT) * 8,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::radix_sort_pairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairs(void * d_temp_storage,
                         size_t& temp_storage_bytes,
                         DoubleBuffer<KeyT>& d_keys,
                         DoubleBuffer<ValueT>& d_values,
                         NumItemsT num_items,
                         int begin_bit = 0,
                         int end_bit = sizeof(KeyT) * 8,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        ::rocprim::double_buffer<ValueT> d_values_db = detail::to_double_buffer(d_values);
        cudaError_t error = (cudaError_t)::rocprim::radix_sort_pairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, d_values_db, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        detail::update_double_buffer(d_values, d_values_db);
        return error;
    }

    template<typename KeyT, typename ValueT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairsDescending(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   const KeyT * d_keys_in,
                                   KeyT * d_keys_out,
                                   const ValueT * d_values_in,
                                   ValueT * d_values_out,
                                   NumItemsT num_items,
                                   int begin_bit = 0,
                                   int end_bit = sizeof(KeyT) * 8,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::radix_sort_pairs_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairsDescending(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   DoubleBuffer<KeyT>& d_keys,
                                   DoubleBuffer<ValueT>& d_values,
                                   NumItemsT num_items,
                                   int begin_bit = 0,
                                   int end_bit = sizeof(KeyT) * 8,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        ::rocprim::double_buffer<ValueT> d_values_db = detail::to_double_buffer(d_values);
        cudaError_t error = (cudaError_t)::rocprim::radix_sort_pairs_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, d_values_db, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        detail::update_double_buffer(d_values, d_values_db);
        return error;
    }

    template<typename KeyT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeys(void * d_temp_storage,
                        size_t& temp_storage_bytes,
                        const KeyT * d_keys_in,
                        KeyT * d_keys_out,
                        NumItemsT num_items,
                        int begin_bit = 0,
                        int end_bit = sizeof(KeyT) * 8,
                        cudaStream_t stream = 0,
                        bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::radix_sort_keys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeys(void * d_temp_storage,
                        size_t& temp_storage_bytes,
                        DoubleBuffer<KeyT>& d_keys,
                        NumItemsT num_items,
                        int begin_bit = 0,
                        int end_bit = sizeof(KeyT) * 8,
                        cudaStream_t stream = 0,
                        bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        cudaError_t error = (cudaError_t)::rocprim::radix_sort_keys(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        return error;
    }

    template<typename KeyT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeysDescending(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  const KeyT * d_keys_in,
                                  KeyT * d_keys_out,
                                  NumItemsT num_items,
                                  int begin_bit = 0,
                                  int end_bit = sizeof(KeyT) * 8,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::radix_sort_keys_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename NumItemsT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeysDescending(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  DoubleBuffer<KeyT>& d_keys,
                                  NumItemsT num_items,
                                  int begin_bit = 0,
                                  int end_bit = sizeof(KeyT) * 8,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        cudaError_t error = (cudaError_t)::rocprim::radix_sort_keys_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, num_items,
            begin_bit, end_bit,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        return error;
    }
};

END_HIPCUB_NAMESPACE

#endif
