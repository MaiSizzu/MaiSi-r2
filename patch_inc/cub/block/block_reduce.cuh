

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_

#include <type_traits>
#include "../config.hpp"

#include <cub/rocprim/block/block_reduce.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
    inline constexpr
    typename std::underlying_type<::rocprim::block_reduce_algorithm>::type
    to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm v)
    {
        using utype = std::underlying_type<::rocprim::block_reduce_algorithm>::type;
        return static_cast<utype>(v);
    }
}

enum BlockReduceAlgorithm
{
    BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
        = detail::to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm::raking_reduce_commutative_only),
    BLOCK_REDUCE_RAKING
        = detail::to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm::raking_reduce),
    BLOCK_REDUCE_WARP_REDUCTIONS
        = detail::to_BlockReduceAlgorithm_enum(::rocprim::block_reduce_algorithm::using_warp_reduce)
};

template<
    typename T,
    int BLOCK_DIM_X,
    BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockReduce
    : private ::rocprim::block_reduce<
        T,
        BLOCK_DIM_X,
        static_cast<::rocprim::block_reduce_algorithm>(ALGORITHM),
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_reduce<
            T,
            BLOCK_DIM_X,
            static_cast<::rocprim::block_reduce_algorithm>(ALGORITHM),
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockReduce() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockReduce(TempStorage& temp_storage) : temp_storage_(temp_storage)
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

    template<int ITEMS_PER_THREAD>
    HIPCUB_DEVICE inline
    T Sum(T(&input)[ITEMS_PER_THREAD])
    {
        T output;
        base_type::reduce(input, output, temp_storage_);
        return output;
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

    template<int ITEMS_PER_THREAD, typename ReduceOp>
    HIPCUB_DEVICE inline
    T Reduce(T(&input)[ITEMS_PER_THREAD], ReduceOp reduce_op)
    {
        T output;
        base_type::reduce(input, output, temp_storage_, reduce_op);
        return output;
    }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif
