

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_MERGE_SORT_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_MERGE_SORT_HPP_

#include "../config.hpp"

#include "../util_type.cuh"

#include <cub/rocprim/device/device_merge_sort.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DeviceMergeSort
{
    template<typename KeyIteratorT, typename ValueIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static cudaError_t SortPairs(void *         d_temp_storage,
                                                        std::size_t &  temp_storage_bytes,
                                                        KeyIteratorT   d_keys,
                                                        ValueIteratorT d_items,
                                                        OffsetT        num_items,
                                                        CompareOpT     compare_op,
                                                        cudaStream_t    stream            = 0,
                                                        bool           debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::merge_sort(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys,
                                     d_keys,
                                     d_items,
                                     d_items,
                                     num_items,
                                     compare_op,
                                     stream,
                                     debug_synchronous);
    }

    template<typename KeyInputIteratorT,
             typename ValueInputIteratorT,
             typename KeyIteratorT,
             typename ValueIteratorT,
             typename OffsetT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static cudaError_t SortPairsCopy(void *              d_temp_storage,
                                                            std::size_t &       temp_storage_bytes,
                                                            KeyInputIteratorT   d_input_keys,
                                                            ValueInputIteratorT d_input_items,
                                                            KeyIteratorT        d_output_keys,
                                                            ValueIteratorT      d_output_items,
                                                            OffsetT             num_items,
                                                            CompareOpT          compare_op,
                                                            cudaStream_t         stream = 0,
                                                            bool debug_synchronous     = false)
    {
        return (cudaError_t)::rocprim::merge_sort(d_temp_storage,
                                     temp_storage_bytes,
                                     d_input_keys,
                                     d_output_keys,
                                     d_input_items,
                                     d_output_items,
                                     num_items,
                                     compare_op,
                                     stream,
                                     debug_synchronous);
    }

    template<typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static cudaError_t SortKeys(void *        d_temp_storage,
                                                       std::size_t & temp_storage_bytes,
                                                       KeyIteratorT  d_keys,
                                                       OffsetT       num_items,
                                                       CompareOpT    compare_op,
                                                       cudaStream_t   stream            = 0,
                                                       bool          debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::merge_sort(
            d_temp_storage, temp_storage_bytes,
            d_keys, d_keys, num_items,
            compare_op, stream, debug_synchronous
        );
    }

    template<typename KeyInputIteratorT,
             typename KeyIteratorT,
             typename OffsetT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static cudaError_t SortKeysCopy(void * d_temp_storage,
                                                           std::size_t & temp_storage_bytes,
                                                           KeyInputIteratorT d_input_keys,
                                                           KeyIteratorT      d_output_keys,
                                                           OffsetT           num_items,
                                                           CompareOpT        compare_op,
                                                           cudaStream_t stream = 0,
                                                           bool debug_synchronous = false)

    {
        return (cudaError_t)::rocprim::merge_sort(
            d_temp_storage, temp_storage_bytes,
            d_input_keys, d_output_keys, num_items,
            compare_op, stream, debug_synchronous
            );
    }

    template <typename KeyIteratorT,
             typename ValueIteratorT,
             typename OffsetT,
             typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static cudaError_t
    StableSortPairs(void *d_temp_storage,
                    std::size_t &temp_storage_bytes,
                    KeyIteratorT d_keys,
                    ValueIteratorT d_items,
                    OffsetT num_items,
                    CompareOpT compare_op,
                    cudaStream_t stream = 0,
                    bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::merge_sort(d_temp_storage,
                                     temp_storage_bytes,
                                     d_keys,
                                     d_keys,
                                     d_items,
                                     d_items,
                                     num_items,
                                     compare_op,
                                     stream,
                                     debug_synchronous);
    }

    template<typename KeyIteratorT, typename OffsetT, typename CompareOpT>
    HIPCUB_RUNTIME_FUNCTION static cudaError_t StableSortKeys(void *        d_temp_storage,
                                                             std::size_t & temp_storage_bytes,
                                                             KeyIteratorT  d_keys,
                                                             OffsetT       num_items,
                                                             CompareOpT    compare_op,
                                                             cudaStream_t   stream   = 0,
                                                             bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::merge_sort(
            d_temp_storage, temp_storage_bytes,
            d_keys, d_keys, num_items,
            compare_op, stream, debug_synchronous
        );
    }

};
END_HIPCUB_NAMESPACE

#endif
