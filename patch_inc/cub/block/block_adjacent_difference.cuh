

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_

#include "../config.hpp"

#include <cub/rocprim/block/block_adjacent_difference.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

  template<class T, class FlagOp, class = void>
  struct WithBIndexArg
      : std::false_type
  { };

  template<class T, class FlagOp>
  struct WithBIndexArg<
          T, FlagOp,
          typename std::conditional<
             true,
             void,
             decltype(std::declval<FlagOp>()(std::declval<T>(), std::declval<T>(), 0))
          >::type
      > : std::true_type
  { };

}

template<
    typename T,
    int BLOCK_DIM_X,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockAdjacentDifference
    : private ::rocprim::block_adjacent_difference<
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
        typename ::rocprim::block_adjacent_difference<
            T,
            BLOCK_DIM_X,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockAdjacentDifference() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockAdjacentDifference(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_heads(head_flags, input, flag_op, temp_storage_);
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagHeads(FlagT (&head_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op,
                   T tile_predecessor_item)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_heads(head_flags, tile_predecessor_item, input, flag_op, temp_storage_);
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_tails(tail_flags, input, flag_op, temp_storage_);
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagTails(FlagT (&tail_flags)[ITEMS_PER_THREAD],
                   T (&input)[ITEMS_PER_THREAD],
                   FlagOp flag_op,
                   T tile_successor_item)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_tails(tail_flags, tile_successor_item, input, flag_op, temp_storage_);
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_heads_and_tails(
            head_flags, tail_flags, input,
            flag_op, temp_storage_
        );
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T tile_successor_item,
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_heads_and_tails(
            head_flags, tail_flags, tile_successor_item, input,
            flag_op, temp_storage_
        );
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           T tile_predecessor_item,
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, input,
            flag_op, temp_storage_
        );
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template<int ITEMS_PER_THREAD, typename FlagT, typename FlagOp>
    [[deprecated("The Flags API of BlockAdjacentDifference is deprecated.")]]
    HIPCUB_DEVICE inline
    void FlagHeadsAndTails(FlagT (&head_flags)[ITEMS_PER_THREAD],
                           T tile_predecessor_item,
                           FlagT (&tail_flags)[ITEMS_PER_THREAD],
                           T tile_successor_item,
                           T (&input)[ITEMS_PER_THREAD],
                           FlagOp flag_op)
    {
        HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH("-Wdeprecated")
        base_type::flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item, input,
            flag_op, temp_storage_
        );
        HIPCUB_CLANG_SUPPRESS_WARNING_POP
    }

    template <int ITEMS_PER_THREAD, typename OutputType, typename DifferenceOpT>
    HIPCUB_DEVICE inline
    void SubtractLeft(T (&input)[ITEMS_PER_THREAD],
                      OutputType (&output)[ITEMS_PER_THREAD],
                      DifferenceOpT difference_op)
    {
        base_type::subtract_left(
            input, output, difference_op, temp_storage_
        );
    }

    template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
    HIPCUB_DEVICE inline
    void SubtractLeft(T (&input)[ITEMS_PER_THREAD],
                      OutputT (&output)[ITEMS_PER_THREAD],
                      DifferenceOpT difference_op,
                      T tile_predecessor_item)
    {
        base_type::subtract_left(
            input, output, difference_op, tile_predecessor_item, temp_storage_
        );
    }

    template <int ITEMS_PER_THREAD, typename OutputType, typename DifferenceOpT>
    HIPCUB_DEVICE inline
    void SubtractLeftPartialTile(T (&input)[ITEMS_PER_THREAD],
                                OutputType (&output)[ITEMS_PER_THREAD],
                                DifferenceOpT difference_op,
                                int valid_items)
    {
        base_type::subtract_left_partial(
            input, output, difference_op, valid_items, temp_storage_
        );
    }

    template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
    HIPCUB_DEVICE inline
    void SubtractRight(T (&input)[ITEMS_PER_THREAD],
                      OutputT (&output)[ITEMS_PER_THREAD],
                      DifferenceOpT difference_op)
    {
        base_type::subtract_right(
            input, output, difference_op, temp_storage_
        );
    }

    template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
    HIPCUB_DEVICE inline
    void SubtractRight(T (&input)[ITEMS_PER_THREAD],
                      OutputT (&output)[ITEMS_PER_THREAD],
                      DifferenceOpT difference_op,
                      T tile_successor_item)
    {
        base_type::subtract_right(
            input, output, difference_op, tile_successor_item, temp_storage_
        );
    }

    template <int ITEMS_PER_THREAD, typename OutputT, typename DifferenceOpT>
    HIPCUB_DEVICE inline
    void SubtractRightPartialTile(T (&input)[ITEMS_PER_THREAD],
                                  OutputT (&output)[ITEMS_PER_THREAD],
                                  DifferenceOpT difference_op,
                                  int valid_items)
    {
        base_type::subtract_right_partial(
            input, output, difference_op, valid_items, temp_storage_
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
