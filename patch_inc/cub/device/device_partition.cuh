

#ifndef HIPCUB_ROCPRIM_DEVICE_PARTITION_HPP_
#define HIPCUB_ROCPRIM_DEVICE_PARTITION_HPP_

#include "../config.hpp"

#include <cub/rocprim/device/device_partition.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DevicePartition
{
    template <
        typename                    InputIteratorT,
        typename                    FlagIterator,
        typename                    OutputIteratorT,
        typename                    NumSelectedIteratorT>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t Flagged(
        void*                       d_temp_storage,
        size_t                      &temp_storage_bytes,
        InputIteratorT              d_in,
        FlagIterator                d_flags,
        OutputIteratorT             d_out,
        NumSelectedIteratorT        d_num_selected_out,
        int                         num_items,
        cudaStream_t                 stream             = 0,
        bool                        debug_synchronous  = false)
    {
        return (cudaError_t)rocprim::partition(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_flags,
            d_out,
            d_num_selected_out,
            num_items,
            stream,
            debug_synchronous);
    }

    template <
        typename                    InputIteratorT,
        typename                    OutputIteratorT,
        typename                    NumSelectedIteratorT,
        typename                    SelectOp>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t If(
        void*                       d_temp_storage,
        size_t                      &temp_storage_bytes,
        InputIteratorT              d_in,
        OutputIteratorT             d_out,
        NumSelectedIteratorT        d_num_selected_out,
        int                         num_items,
        SelectOp                    select_op,
        cudaStream_t                 stream             = 0,
        bool                        debug_synchronous  = false)
    {
        return (cudaError_t)rocprim::partition(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            d_num_selected_out,
            num_items,
            select_op,
            stream,
            debug_synchronous);
    }
    
    template <typename InputIteratorT,
              typename FirstOutputIteratorT,
              typename SecondOutputIteratorT,
              typename UnselectedOutputIteratorT,
              typename NumSelectedIteratorT,
              typename SelectFirstPartOp,
              typename SelectSecondPartOp>
    HIPCUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t
    If(void *d_temp_storage,
       std::size_t &temp_storage_bytes,
       InputIteratorT d_in,
       FirstOutputIteratorT d_first_part_out,
       SecondOutputIteratorT d_second_part_out,
       UnselectedOutputIteratorT d_unselected_out,
       NumSelectedIteratorT d_num_selected_out,
       int num_items,
       SelectFirstPartOp select_first_part_op,
       SelectSecondPartOp select_second_part_op,
       cudaStream_t stream     = 0,
       bool debug_synchronous = false)
    {
        return (cudaError_t)rocprim::partition_three_way(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_first_part_out,
            d_second_part_out,
            d_unselected_out,
            d_num_selected_out,
            num_items,
            select_first_part_op,
            select_second_part_op,
            stream,
            debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
