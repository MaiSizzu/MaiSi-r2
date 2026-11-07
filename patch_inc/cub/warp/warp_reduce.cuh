

#ifndef HIPCUB_ROCPRIM_WARP_WARP_REDUCE_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_REDUCE_HPP_

#include "../config.hpp"

#include "../util_ptx.cuh"
#include "../thread/thread_operators.cuh"

#include <cub/rocprim/warp/warp_reduce.hpp>

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH>
class WarpReduce : private ::rocprim::warp_reduce<T, LOGICAL_WARP_THREADS>
{
    static_assert(LOGICAL_WARP_THREADS > 0, "LOGICAL_WARP_THREADS must be greater than 0");
    using base_type = typename ::rocprim::warp_reduce<T, LOGICAL_WARP_THREADS>;

    typename base_type::storage_type &temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    WarpReduce(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    T Sum(T input)
    {
        base_type::reduce(input, input, temp_storage_);
        return input;
    }

    HIPCUB_DEVICE inline
    T Sum(T input, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_);
        return input;
    }

    template<typename FlagT>
    HIPCUB_DEVICE inline
    T HeadSegmentedSum(T input, FlagT head_flag)
    {
        base_type::head_segmented_reduce(input, input, head_flag, temp_storage_);
        return input;
    }

    template<typename FlagT>
    HIPCUB_DEVICE inline
    T TailSegmentedSum(T input, FlagT tail_flag)
    {
        base_type::tail_segmented_reduce(input, input, tail_flag, temp_storage_);
        return input;
    }

    template<typename ReduceOp>
    HIPCUB_DEVICE inline
    T Reduce(T input, ReduceOp reduce_op)
    {
        base_type::reduce(input, input, temp_storage_, reduce_op);
        return input;
    }

    template<typename ReduceOp>
    HIPCUB_DEVICE inline
    T Reduce(T input, ReduceOp reduce_op, int valid_items)
    {
        base_type::reduce(input, input, valid_items, temp_storage_, reduce_op);
        return input;
    }

    template<typename ReduceOp, typename FlagT>
    HIPCUB_DEVICE inline
    T HeadSegmentedReduce(T input, FlagT head_flag, ReduceOp reduce_op)
    {
        base_type::head_segmented_reduce(
            input, input, head_flag, temp_storage_, reduce_op
        );
        return input;
    }

    template<typename ReduceOp, typename FlagT>
    HIPCUB_DEVICE inline
    T TailSegmentedReduce(T input, FlagT tail_flag, ReduceOp reduce_op)
    {
        base_type::tail_segmented_reduce(
            input, input, tail_flag, temp_storage_, reduce_op
        );
        return input;
    }
};

END_HIPCUB_NAMESPACE

#endif
