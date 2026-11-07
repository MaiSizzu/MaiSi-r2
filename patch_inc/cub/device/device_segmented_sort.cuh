

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SORT_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_SORT_HPP_

#include "../config.hpp"

#include "../util_type.cuh"

#include <cub/rocprim/device/device_segmented_radix_sort.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DeviceSegmentedSort
{
    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairs(void * d_temp_storage,
                         size_t& temp_storage_bytes,
                         const KeyT * d_keys_in,
                         KeyT * d_keys_out,
                         const ValueT * d_values_in,
                         ValueT * d_values_out,
                         int num_items,
                         int num_segments,
                         OffsetIteratorT d_begin_offsets,
                         OffsetIteratorT d_end_offsets,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::segmented_radix_sort_pairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairs(void * d_temp_storage,
                         size_t& temp_storage_bytes,
                         DoubleBuffer<KeyT>& d_keys,
                         DoubleBuffer<ValueT>& d_values,
                         int num_items,
                         int num_segments,
                         OffsetIteratorT d_begin_offsets,
                         OffsetIteratorT d_end_offsets,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        ::rocprim::double_buffer<ValueT> d_values_db = detail::to_double_buffer(d_values);
        cudaError_t error = (cudaError_t)::rocprim::segmented_radix_sort_pairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, d_values_db, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        detail::update_double_buffer(d_values, d_values_db);
        return error;
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairsDescending(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   const KeyT * d_keys_in,
                                   KeyT * d_keys_out,
                                   const ValueT * d_values_in,
                                   ValueT * d_values_out,
                                   int num_items,
                                   int num_segments,
                                   OffsetIteratorT d_begin_offsets,
                                   OffsetIteratorT d_end_offsets,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::segmented_radix_sort_pairs_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortPairsDescending(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   DoubleBuffer<KeyT>& d_keys,
                                   DoubleBuffer<ValueT>& d_values,
                                   int num_items,
                                   int num_segments,
                                   OffsetIteratorT d_begin_offsets,
                                   OffsetIteratorT d_end_offsets,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        ::rocprim::double_buffer<ValueT> d_values_db = detail::to_double_buffer(d_values);
        cudaError_t error = (cudaError_t)::rocprim::segmented_radix_sort_pairs_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, d_values_db, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        detail::update_double_buffer(d_values, d_values_db);
        return error;
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeys(void * d_temp_storage,
                        size_t& temp_storage_bytes,
                        const KeyT * d_keys_in,
                        KeyT * d_keys_out,
                        int num_items,
                        int num_segments,
                        OffsetIteratorT d_begin_offsets,
                        OffsetIteratorT d_end_offsets,
                        cudaStream_t stream = 0,
                        bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::segmented_radix_sort_keys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeys(void * d_temp_storage,
                        size_t& temp_storage_bytes,
                        DoubleBuffer<KeyT>& d_keys,
                        int num_items,
                        int num_segments,
                        OffsetIteratorT d_begin_offsets,
                        OffsetIteratorT d_end_offsets,
                        cudaStream_t stream = 0,
                        bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        cudaError_t error = (cudaError_t)::rocprim::segmented_radix_sort_keys(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        return error;
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeysDescending(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  const KeyT * d_keys_in,
                                  KeyT * d_keys_out,
                                  int num_items,
                                  int num_segments,
                                  OffsetIteratorT d_begin_offsets,
                                  OffsetIteratorT d_end_offsets,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::segmented_radix_sort_keys_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t SortKeysDescending(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  DoubleBuffer<KeyT>& d_keys,
                                  int num_items,
                                  int num_segments,
                                  OffsetIteratorT d_begin_offsets,
                                  OffsetIteratorT d_end_offsets,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        ::rocprim::double_buffer<KeyT> d_keys_db = detail::to_double_buffer(d_keys);
        cudaError_t error = (cudaError_t)::rocprim::segmented_radix_sort_keys_desc(
            d_temp_storage, temp_storage_bytes,
            d_keys_db, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            0, sizeof(KeyT) * 8,
            stream, debug_synchronous
        );
        detail::update_double_buffer(d_keys, d_keys_db);
        return error;
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortPairs(void * d_temp_storage,
                               size_t& temp_storage_bytes,
                               const KeyT * d_keys_in,
                               KeyT * d_keys_out,
                               const ValueT * d_values_in,
                               ValueT * d_values_out,
                               int num_items,
                               int num_segments,
                               OffsetIteratorT d_begin_offsets,
                               OffsetIteratorT d_end_offsets,
                               cudaStream_t stream = 0,
                               bool debug_synchronous = false)
    {
        return SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortPairs(void * d_temp_storage,
                               size_t& temp_storage_bytes,
                               DoubleBuffer<KeyT>& d_keys,
                               DoubleBuffer<ValueT>& d_values,
                               int num_items,
                               int num_segments,
                               OffsetIteratorT d_begin_offsets,
                               OffsetIteratorT d_end_offsets,
                               cudaStream_t stream = 0,
                               bool debug_synchronous = false)
    {
        return SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys, d_values, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortPairsDescending(void * d_temp_storage,
                                         size_t& temp_storage_bytes,
                                         const KeyT * d_keys_in,
                                         KeyT * d_keys_out,
                                         const ValueT * d_values_in,
                                         ValueT * d_values_out,
                                         int num_items,
                                         int num_segments,
                                         OffsetIteratorT d_begin_offsets,
                                         OffsetIteratorT d_end_offsets,
                                         cudaStream_t stream = 0,
                                         bool debug_synchronous = false)
    {
        return SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename ValueT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortPairsDescending(void * d_temp_storage,
                                         size_t& temp_storage_bytes,
                                         DoubleBuffer<KeyT>& d_keys,
                                         DoubleBuffer<ValueT>& d_values,
                                         int num_items,
                                         int num_segments,
                                         OffsetIteratorT d_begin_offsets,
                                         OffsetIteratorT d_end_offsets,
                                         cudaStream_t stream = 0,
                                         bool debug_synchronous = false)
    {
        return SortPairsDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys, d_values, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortKeys(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              const KeyT * d_keys_in,
                              KeyT * d_keys_out,
                              int num_items,
                              int num_segments,
                              OffsetIteratorT d_begin_offsets,
                              OffsetIteratorT d_end_offsets,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortKeys(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              DoubleBuffer<KeyT>& d_keys,
                              int num_items,
                              int num_segments,
                              OffsetIteratorT d_begin_offsets,
                              OffsetIteratorT d_end_offsets,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortKeysDescending(void * d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        const KeyT * d_keys_in,
                                        KeyT * d_keys_out,
                                        int num_items,
                                        int num_segments,
                                        OffsetIteratorT d_begin_offsets,
                                        OffsetIteratorT d_end_offsets,
                                        cudaStream_t stream = 0,
                                        bool debug_synchronous = false)
    {
        return SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }

    template<typename KeyT, typename OffsetIteratorT>
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t StableSortKeysDescending(void * d_temp_storage,
                                        size_t& temp_storage_bytes,
                                        DoubleBuffer<KeyT>& d_keys,
                                        int num_items,
                                        int num_segments,
                                        OffsetIteratorT d_begin_offsets,
                                        OffsetIteratorT d_end_offsets,
                                        cudaStream_t stream = 0,
                                        bool debug_synchronous = false)
    {
        return SortKeysDescending(
            d_temp_storage, temp_storage_bytes,
            d_keys, num_items,
            num_segments, d_begin_offsets, d_end_offsets,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
