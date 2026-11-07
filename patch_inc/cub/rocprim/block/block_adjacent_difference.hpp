

#ifndef ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_
#define ROCPRIM_BLOCK_BLOCK_ADJACENT_DIFFERENCE_HPP_


#include "detail/block_adjacent_difference_impl.hpp"

#include "../config.hpp"
#include "../detail/various.hpp"






BEGIN_ROCPRIM_NAMESPACE








































template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_adjacent_difference
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::block_adjacent_difference_impl<T, BlockSizeX, BlockSizeY, BlockSizeZ>
#endif
{
    using base_type = detail::block_adjacent_difference_impl<T, BlockSizeX, BlockSizeY, BlockSizeZ>;

    static constexpr unsigned BlockSize = base_type::BlockSize;

    struct storage_type_
    {
        typename base_type::storage_type left;
        typename base_type::storage_type right;
    };

public:









    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_;
    #endif











































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;
        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, head_flags, flag_op, input[0] , storage.get().left);
    }







    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads(head_flags, input, flag_op, storage);
    }




















































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    T tile_predecessor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;
        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, head_flags, flag_op, tile_predecessor_item, storage.get().left);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_left or block_discontinuity.flag_heads instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads(Flag (&head_flags)[ItemsPerThread],
                    T tile_predecessor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads(head_flags, tile_predecessor_item, input, flag_op, storage);
    }












































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags       = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_successor = false;
        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, tail_flags, flag_op, input[0] , storage.get().right);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_tails(tail_flags, input, flag_op, storage);
    }




















































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    T tile_successor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op,
                    storage_type& storage)
    {
        static constexpr auto as_flags       = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_successor = true;
        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use subtract_right or block_discontinuity.flag_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_tails(Flag (&tail_flags)[ItemsPerThread],
                    T tile_successor_item,
                    const T (&input)[ItemsPerThread],
                    FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_tails(tail_flags, tile_successor_item, input, flag_op, storage);
    }












































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;
        static constexpr auto with_successor   = false;


        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, items[0] , storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, items[0] , storage.get().right);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(head_flags, tail_flags, input, flag_op, storage);
    }























































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;
        static constexpr auto with_successor   = true;


        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, items[0] , storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(head_flags, tail_flags, tile_successor_item, input, flag_op, storage);
    }























































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;
        static constexpr auto with_successor   = false;


        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, tile_predecessor_item, storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, items[0] , storage.get().right);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(head_flags, tile_predecessor_item, tail_flags, input, flag_op, storage);
    }





























































    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op,
                              storage_type& storage)
    {
        static constexpr auto as_flags         = true;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;
        static constexpr auto with_successor   = true;


        T items[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; ++i) {
            items[i] = input[i];
        }

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            items, head_flags, flag_op, tile_predecessor_item, storage.get().left);

        base_type::template apply_right<as_flags, reversed, with_successor>(
            items, tail_flags, flag_op, tile_successor_item, storage.get().right);
    }








    template<unsigned int ItemsPerThread, class Flag, class FlagOp>
    [[deprecated("The flags API of block_adjacent_difference is deprecated."
                 "Use block_discontinuity.flag_heads_and_tails instead.")]]
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void flag_heads_and_tails(Flag (&head_flags)[ItemsPerThread],
                              T tile_predecessor_item,
                              Flag (&tail_flags)[ItemsPerThread],
                              T tile_successor_item,
                              const T (&input)[ItemsPerThread],
                              FlagOp flag_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        flag_heads_and_tails(
            head_flags, tile_predecessor_item, tail_flags, tile_successor_item,
            input, flag_op, storage
        );
    }
























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left(const T (&input)[ItemsPerThread],
                                                     Output (&output)[ItemsPerThread],
                                                     const BinaryFunction op,
                                                     storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, output, op, input[0] , storage.get().left);
    }



























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left(const T (&input)[ItemsPerThread],
                                                     Output (&output)[ItemsPerThread],
                                                     const BinaryFunction op,
                                                     const T              tile_predecessor,
                                                     storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;

        base_type::template apply_left<as_flags, reversed, with_predecessor>(
            input, output, op, tile_predecessor, storage.get().left);
    }



























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left_partial(const T (&input)[ItemsPerThread],
                                                             Output (&output)[ItemsPerThread],
                                                             const BinaryFunction op,
                                                             const unsigned int   valid_items,
                                                             storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = false;

        base_type::template apply_left_partial<as_flags, reversed, with_predecessor>(
            input, output, op, input[0] , valid_items, storage.get().left);
    }























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_left_partial(const T (&input)[ItemsPerThread],
                                                             Output (&output)[ItemsPerThread],
                                                             const BinaryFunction op,
                                                             const T              tile_predecessor,
                                                             const unsigned int   valid_items,
                                                             storage_type&        storage)
    {
        static constexpr auto as_flags         = false;
        static constexpr auto reversed         = true;
        static constexpr auto with_predecessor = true;

        base_type::template apply_left_partial<as_flags, reversed, with_predecessor>(
            input, output, op, tile_predecessor, valid_items, storage.get().left);
    }
























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_right(const T (&input)[ItemsPerThread],
                                                      Output (&output)[ItemsPerThread],
                                                      const BinaryFunction op,
                                                      storage_type&        storage)
    {
        static constexpr auto as_flags       = false;
        static constexpr auto reversed       = false;
        static constexpr auto with_successor = false;

        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, output, op, input[0] , storage.get().right);
    }




























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_right(const T (&input)[ItemsPerThread],
                                                      Output (&output)[ItemsPerThread],
                                                      const BinaryFunction op,
                                                      const T              tile_successor,
                                                      storage_type&        storage)
    {
        static constexpr auto as_flags       = false;
        static constexpr auto reversed       = false;
        static constexpr auto with_successor = true;

        base_type::template apply_right<as_flags, reversed, with_successor>(
            input, output, op, tile_successor, storage.get().right);
    }


























    template <typename Output, unsigned int ItemsPerThread, typename BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE void subtract_right_partial(const T (&input)[ItemsPerThread],
                                                              Output (&output)[ItemsPerThread],
                                                              const BinaryFunction op,
                                                              const unsigned int   valid_items,
                                                              storage_type&        storage)
    {
        static constexpr auto as_flags = false;
        static constexpr auto reversed = false;

        base_type::template apply_right_partial<as_flags, reversed>(
            input, output, op, valid_items, storage.get().right);
    }
};

END_ROCPRIM_NAMESPACE




#endif
