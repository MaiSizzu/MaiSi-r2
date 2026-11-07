



















#ifndef ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
#define ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE




















template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread])
{
    unsigned int offset = flat_id * ItemsPerThread;
    InputIterator thread_iter = block_input + offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item];
    }
}


















template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid)
{
    unsigned int offset = flat_id * ItemsPerThread;
    InputIterator thread_iter = block_input + offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        if (item + offset < valid)
        {
            items[item] = thread_iter[item];
        }
    }
}





















template<
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_blocked(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid,
                               Default out_of_bounds)
{
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = static_cast<T>(out_of_bounds);
    }


    block_load_direct_blocked(flat_id, block_input, items, valid);
}


























template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_load_direct_blocked_vectorized(unsigned int flat_id,
                                     T* block_input,
                                     U (&items)[ItemsPerThread]) -> typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    typedef typename detail::match_vector_type<T, ItemsPerThread>::type vector_type;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(vector_type);
    vector_type vector_items[vectors_per_thread];

    const vector_type* vector_ptr = reinterpret_cast<const vector_type*>(block_input) +
        (flat_id * vectors_per_thread);

    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < vectors_per_thread; item++)
    {
        vector_items[item] = *(vector_ptr + item);
    }

    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = *(reinterpret_cast<T*>(vector_items) + item);
    }
}

template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_load_direct_blocked_vectorized(unsigned int flat_id,
                                     T* block_input,
                                     U (&items)[ItemsPerThread]) -> typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    block_load_direct_blocked(flat_id, block_input, items);
}


















template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread])
{
    InputIterator thread_iter = block_input + flat_id;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * BlockSize];
    }
}



















template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid)
{
    InputIterator thread_iter = block_input + flat_id;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
            items[item] = thread_iter[offset];
        }
    }
}






















template<
    unsigned int BlockSize,
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_striped(unsigned int flat_id,
                               InputIterator block_input,
                               T (&items)[ItemsPerThread],
                               unsigned int valid,
                               Default out_of_bounds)
{
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
}

























template<
    unsigned int WarpSize = device_warp_size(),
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread])
{
    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= device_warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;

    InputIterator thread_iter = block_input + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = thread_iter[item * WarpSize];
    }
}


























template<
    unsigned int WarpSize = device_warp_size(),
    class InputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid)
{
    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= device_warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;

    InputIterator thread_iter = block_input + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * WarpSize;
        if (warp_offset + thread_id + offset < valid)
        {
            items[item] = thread_iter[offset];
        }
    }
}





























template<
    unsigned int WarpSize = device_warp_size(),
    class InputIterator,
    class T,
    unsigned int ItemsPerThread,
    class Default
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_load_direct_warp_striped(unsigned int flat_id,
                                    InputIterator block_input,
                                    T (&items)[ItemsPerThread],
                                    unsigned int valid,
                                    Default out_of_bounds)
{
    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= device_warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        items[item] = out_of_bounds;
    }

    block_load_direct_warp_striped<WarpSize>(flat_id, block_input, items, valid);
}

END_ROCPRIM_NAMESPACE




#endif
