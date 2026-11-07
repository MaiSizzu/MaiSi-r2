



















#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_ADJACENT_DIFFERENCE_IMPL_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_ADJACENT_DIFFERENCE_IMPL_HPP_

#include "../../config.hpp"
#include "../../detail/various.hpp"
#include "../../intrinsics/thread.hpp"

#include <type_traits>

#include <cassert>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{







template <class T, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE auto apply(BinaryFunction op,
                                         const T&       a,
                                         const T&       b,
                                         unsigned int   index,
                                         bool_constant<true> ,
                                         bool_constant<false> ) -> decltype(op(b, a, index))
{
    return op(a, b, index);
}

template <class T, class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE auto apply(BinaryFunction op,
                                         const T&       a,
                                         const T&       b,
                                         unsigned int   index,
                                         bool_constant<true> ,
                                         bool_constant<true> )
    -> decltype(op(b, a, index))
{
    return op(b, a, index);
}

template <typename T, typename BinaryFunction, bool AsFlags>
ROCPRIM_DEVICE ROCPRIM_INLINE auto apply(BinaryFunction op,
                                         const T&       a,
                                         const T&       b,
                                         unsigned int,
                                         bool_constant<AsFlags> ,
                                         bool_constant<false> ) -> decltype(op(b, a))
{
    return op(a, b);
}

template <typename T, typename BinaryFunction, bool AsFlags>
ROCPRIM_DEVICE ROCPRIM_INLINE auto apply(BinaryFunction op,
                                         const T&       a,
                                         const T&       b,
                                         unsigned int,
                                         bool_constant<AsFlags> ,
                                         bool_constant<true> ) -> decltype(op(b, a))
{
    return op(b, a);
}

template <typename T,
          unsigned int BlockSizeX,
          unsigned int BlockSizeY = 1,
          unsigned int BlockSizeZ = 1>
class block_adjacent_difference_impl
{
public:
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    struct storage_type
    {
        T items[BlockSize];
    };

    template <bool         AsFlags,
              bool         Reversed,
              bool         WithTilePredecessor,
              unsigned int ItemsPerThread,
              typename Output,
              typename BinaryFunction>
    ROCPRIM_DEVICE void apply_left(const T (&input)[ItemsPerThread],
                                   Output (&output)[ItemsPerThread],
                                   BinaryFunction op,
                                   const T        tile_predecessor_item,
                                   storage_type&  storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags> {};
        static constexpr auto reversed = bool_constant<Reversed> {};

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();


        storage.items[flat_id] = input[ItemsPerThread - 1];

        ROCPRIM_UNROLL
        for(unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            output[i] = detail::apply(
                op, input[i - 1], input[i], flat_id * ItemsPerThread + i, as_flags, reversed);
        }
        ::rocprim::syncthreads();

        if ROCPRIM_IF_CONSTEXPR (WithTilePredecessor)
        {
            T predecessor_item = tile_predecessor_item;
            if(flat_id != 0) {
                predecessor_item = storage.items[flat_id - 1];
            }

            output[0] = detail::apply(
                op, predecessor_item, input[0], flat_id * ItemsPerThread, as_flags, reversed);
        }
        else
        {
            output[0] = get_default_item(input, 0, as_flags);
            if(flat_id != 0) {
                output[0] = detail::apply(op,
                                          storage.items[flat_id - 1],
                                          input[0],
                                          flat_id * ItemsPerThread,
                                          as_flags,
                                          reversed);
            }
        }
    }

    template <bool         AsFlags,
              bool         Reversed,
              bool         WithTilePredecessor,
              unsigned int ItemsPerThread,
              typename Output,
              typename BinaryFunction>
    ROCPRIM_DEVICE void apply_left_partial(const T (&input)[ItemsPerThread],
                                           Output (&output)[ItemsPerThread],
                                           BinaryFunction     op,
                                           const T            tile_predecessor_item,
                                           const unsigned int valid_items,
                                           storage_type&      storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags> {};
        static constexpr auto reversed = bool_constant<Reversed> {};

        assert(valid_items <= BlockSize * ItemsPerThread);

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();


        storage.items[flat_id] = input[ItemsPerThread - 1];

        ROCPRIM_UNROLL
        for(unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            const unsigned int index = flat_id * ItemsPerThread + i;
            output[i] = get_default_item(input, i, as_flags);
            if(index < valid_items) {
                output[i] = detail::apply(op, input[i - 1], input[i], index, as_flags, reversed);
            }
        }
        ::rocprim::syncthreads();

        const unsigned int index = flat_id * ItemsPerThread;

        if ROCPRIM_IF_CONSTEXPR (WithTilePredecessor)
        {
            T predecessor_item = tile_predecessor_item;
            if(flat_id != 0) {
                predecessor_item = storage.items[flat_id - 1];
            }

            output[0] = get_default_item(input, 0, as_flags);
            if(index < valid_items)
            {
                output[0]
                    = detail::apply(op, predecessor_item, input[0], index, as_flags, reversed);
            }
        }
        else
        {
            output[0] = get_default_item(input, 0, as_flags);
            if(flat_id != 0 && index < valid_items)
            {
                output[0] = detail::apply(op,
                                          storage.items[flat_id - 1],
                                          input[0],
                                          flat_id * ItemsPerThread,
                                          as_flags,
                                          reversed);
            }
        }
    }

    template <bool         AsFlags,
              bool         Reversed,
              bool         WithTileSuccessor,
              unsigned int ItemsPerThread,
              typename Output,
              typename BinaryFunction>
    ROCPRIM_DEVICE void apply_right(const T (&input)[ItemsPerThread],
                                    Output (&output)[ItemsPerThread],
                                    BinaryFunction op,
                                    const T        tile_successor_item,
                                    storage_type&  storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags> {};
        static constexpr auto reversed = bool_constant<Reversed> {};

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();


        storage.items[flat_id] = input[0];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread - 1; ++i)
        {
            output[i] = detail::apply(
                op, input[i], input[i + 1], flat_id * ItemsPerThread + i + 1, as_flags, reversed);
        }
        ::rocprim::syncthreads();

        if ROCPRIM_IF_CONSTEXPR (WithTileSuccessor)
        {
            T successor_item = tile_successor_item;
            if(flat_id != BlockSize - 1) {
                successor_item = storage.items[flat_id + 1];
            }

            output[ItemsPerThread - 1] = detail::apply(op,
                                                       input[ItemsPerThread - 1],
                                                       successor_item,
                                                       flat_id * ItemsPerThread + ItemsPerThread,
                                                       as_flags,
                                                       reversed);
        }
        else
        {
            output[ItemsPerThread - 1] = get_default_item(input, ItemsPerThread - 1, as_flags);
            if(flat_id != BlockSize - 1) {
                output[ItemsPerThread - 1]
                    = detail::apply(op,
                                    input[ItemsPerThread - 1],
                                    storage.items[flat_id + 1],
                                    flat_id * ItemsPerThread + ItemsPerThread,
                                    as_flags,
                                    reversed);
            }
        }
    }
    template <bool         AsFlags,
              bool         Reversed,
              unsigned int ItemsPerThread,
              typename Output,
              typename BinaryFunction>
    ROCPRIM_DEVICE void apply_right_partial(const T (&input)[ItemsPerThread],
                                            Output (&output)[ItemsPerThread],
                                            BinaryFunction     op,
                                            const unsigned int valid_items,
                                            storage_type&      storage)
    {
        static constexpr auto as_flags = bool_constant<AsFlags> {};
        static constexpr auto reversed = bool_constant<Reversed> {};

        assert(valid_items <= BlockSize * ItemsPerThread);

        const unsigned int flat_id
            = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();


        storage.items[flat_id] = input[0];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread - 1; ++i)
        {
            const unsigned int index = flat_id * ItemsPerThread + i + 1;
            output[i] = get_default_item(input, i, as_flags);
            if(index < valid_items)
            {
                output[i] = detail::apply(op, input[i], input[i + 1], index, as_flags, reversed);
            }
        }
        ::rocprim::syncthreads();

        output[ItemsPerThread - 1] = get_default_item(input, ItemsPerThread - 1, as_flags);

        const unsigned int next_thread_index = flat_id * ItemsPerThread + ItemsPerThread;
        if(next_thread_index < valid_items)
        {
            output[ItemsPerThread - 1] = detail::apply(op,
                                                       input[ItemsPerThread - 1],
                                                       storage.items[flat_id + 1],
                                                       next_thread_index,
                                                       as_flags,
                                                       reversed);
        }
    }

private:
    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE int get_default_item(const T (&)[ItemsPerThread],
                                        unsigned int ,
                                        bool_constant<true> )
    {
        return 1;
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE T get_default_item(const T (&input)[ItemsPerThread],
                                      const unsigned int index,
                                      bool_constant<false> )
    {
        return input[index];
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
