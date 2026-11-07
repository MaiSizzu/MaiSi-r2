

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_DISCONTINUITY_HPP_

#include "../config.hpp"

#include <cub/rocprim/block/block_discontinuity.hpp>

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int BLOCK_DIM_X,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockDiscontinuity
    : private ::rocprim::block_discontinuity<
        T,
        BLOCK_DIM_X,
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_discontinuity<
            T,
            BLOCK_DIM_X,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockDiscontinuity() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockDiscontinuity(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op)
    {
        base_type::flag_heads(head_flags, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op,
                   T tile_predecessor_item)
    {
        base_type::flag_heads(head_flags, tile_predecessor_item, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op)
    {
        base_type::flag_tails(tail_flags, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op,
                   T tile_successor_item)
    {
        base_type::flag_tails(tail_flags, tile_successor_item, input, flag_op, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tail_flags, input,
            flag_op, temp_storage_
        );
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T tile_successor_item,
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tail_flags, tile_successor_item, input,
            flag_op, temp_storage_
        );
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           T tile_predecessor_item,
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, input,
            flag_op, temp_storage_
        );
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           T tile_predecessor_item,
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T tile_successor_item,
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        base_type::flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input,
            flag_op, temp_storage_
        );
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
