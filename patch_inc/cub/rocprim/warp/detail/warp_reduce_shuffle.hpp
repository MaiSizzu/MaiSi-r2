



















#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHUFFLE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHUFFLE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"
#include "../../detail/various.hpp"

#include "warp_segment_bounds.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    bool UseAllReduce
>
class warp_reduce_shuffle
{
public:
    static_assert(detail::is_power_of_two(WarpSize), "WarpSize must be power of 2");

    using storage_type = detail::empty_storage_type;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, BinaryFunction reduce_op)
    {
        output = input;

        T value;
        ROCPRIM_UNROLL
        for(unsigned int offset = 1; offset < WarpSize; offset *= 2)
        {
            value = warp_shuffle_down(output, offset, WarpSize);
            output = reduce_op(output, value);
        }
        set_output<UseAllReduce>(output);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage;
        this->reduce(input, output, reduce_op);
    }

    template<bool UseAllReduceDummy = UseAllReduce, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, unsigned int valid_items, BinaryFunction reduce_op)
    {
        output = input;

        T value;
        ROCPRIM_UNROLL
        for(unsigned int offset = 1; offset < WarpSize; offset *= 2)
        {
            value = warp_shuffle_down(output, offset, WarpSize);
            unsigned int id = detail::logical_lane_id<WarpSize>();
            if (id + offset < valid_items) output = reduce_op(output, value);
        }
        set_output<UseAllReduceDummy>(output);
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
        this->segmented_reduce<true>(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {
        this->segmented_reduce<false>(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage;
        this->segmented_reduce<true>(input, output, flag, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        (void) storage;
        this->segmented_reduce<false>(input, output, flag, reduce_op);
    }

private:
    template<bool HeadSegmented, class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void segmented_reduce(T input, T& output, Flag flag, BinaryFunction reduce_op)
    {


        auto valid_items_in_segment = last_in_warp_segment<HeadSegmented, WarpSize>(flag) + 1U;
        this->reduce<false>(input, output, valid_items_in_segment, reduce_op);
    }

    template<bool Switch>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Switch == false)>::type
    set_output(T& output)
    {
        (void) output;

    }

    template<bool Switch>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Switch == true)>::type
    set_output(T& output)
    {
        output = warp_shuffle(output, 0, WarpSize);
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
