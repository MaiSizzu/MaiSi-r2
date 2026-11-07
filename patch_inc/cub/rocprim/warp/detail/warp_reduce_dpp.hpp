



















#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_DPP_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"
#include "../../detail/various.hpp"

#include "warp_reduce_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    bool UseAllReduce
>
class warp_reduce_dpp
{
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, BinaryFunction reduce_op)
    {
        output = input;

        if(WarpSize > 1)
        {

            output = reduce_op(warp_move_dpp<T, 0xb1>(output), output);
        }
        if(WarpSize > 2)
        {

            output = reduce_op(warp_move_dpp<T, 0x4e>(output), output);
        }
        if(WarpSize > 4)
        {

            output = reduce_op(warp_move_dpp<T, 0x114>(output), output);
        }
        if(WarpSize > 8)
        {

            output = reduce_op(warp_move_dpp<T, 0x118>(output), output);
        }
#if ROCPRIM_NAVI
        if(WarpSize > 16)
        {

            output = reduce_op(warp_swizzle<T, 0x1e0>(output), output);
        }
#else
        if(WarpSize > 16)
        {

            output = reduce_op(warp_move_dpp<T, 0x142>(output), output);
        }
        if(WarpSize > 32)
        {

            output = reduce_op(warp_move_dpp<T, 0x143>(output), output);
        }
#endif

        output = warp_shuffle(output, WarpSize - 1, WarpSize);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage;
        this->reduce(input, output, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, unsigned int valid_items, BinaryFunction reduce_op)
    {

        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .reduce(input, output, valid_items, reduce_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, unsigned int valid_items,
                storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage;
        this->reduce(input, output, valid_items, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {

        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .head_segmented_reduce(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {

        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .tail_segmented_reduce(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {

        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .head_segmented_reduce(input, output, flag, storage, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {

        warp_reduce_shuffle<T, WarpSize, UseAllReduce>()
            .tail_segmented_reduce(input, output, flag, storage, reduce_op);
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
