



















#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_SHARED_MEM_HPP_

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
class warp_reduce_shared_mem
{
    struct storage_type_
    {
        T values[WarpSize];
    };

public:
    using storage_type = detail::raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, storage_type& storage, BinaryFunction reduce_op)
    {
        constexpr unsigned int ceiling = next_power_of_two(WarpSize);
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        output = input;
        store_volatile(&storage_.values[lid], output);
        ROCPRIM_UNROLL
        for(unsigned int i = ceiling >> 1; i > 0; i >>= 1)
        {
            if (lid + i < WarpSize && lid < i)
            {
                output = load_volatile(&storage_.values[lid]);
                T other = load_volatile(&storage_.values[lid + i]);
                output = reduce_op(output, other);
                store_volatile(&storage_.values[lid], output);
            }
        }
        set_output<UseAllReduce>(output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input, T& output, unsigned int valid_items,
                storage_type& storage, BinaryFunction reduce_op)
    {
        constexpr unsigned int ceiling = next_power_of_two(WarpSize);
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        output = input;
        store_volatile(&storage_.values[lid], output);
        ROCPRIM_UNROLL
        for(unsigned int i = ceiling >> 1; i > 0; i >>= 1)
        {
            if((lid + i) < WarpSize && lid < i && (lid + i) < valid_items)
            {
                output = load_volatile(&storage_.values[lid]);
                T other = load_volatile(&storage_.values[lid + i]);
                output = reduce_op(output, other);
                store_volatile(&storage_.values[lid], output);
            }
        }
        set_output<UseAllReduce>(output, storage);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void head_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        this->segmented_reduce<true>(input, output, flag, storage, reduce_op);
    }

    template<class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void tail_segmented_reduce(T input, T& output, Flag flag,
                               storage_type& storage, BinaryFunction reduce_op)
    {
        this->segmented_reduce<false>(input, output, flag, storage, reduce_op);
    }

private:
    template<bool HeadSegmented, class Flag, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void segmented_reduce(T input, T& output, Flag flag,
                          storage_type& storage, BinaryFunction reduce_op)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        constexpr unsigned int ceiling = next_power_of_two(WarpSize);
        storage_type_& storage_ = storage.get();

        auto last = last_in_warp_segment<HeadSegmented, WarpSize>(flag);

        output = input;
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ceiling; i *= 2)
        {
            store_volatile(&storage_.values[lid], output);
            if((lid + i) <= last)
            {
                T other = load_volatile(&storage_.values[lid + i]);
                output = reduce_op(output, other);
            }
        }
    }

    template<bool Switch>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Switch == false)>::type
    set_output(T& output, storage_type& storage)
    {
        (void) output;
        (void) storage;

    }

    template<bool Switch>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    typename std::enable_if<(Switch == true)>::type
    set_output(T& output, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        output = load_volatile(&storage_.values[0]);
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
