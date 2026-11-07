



















#ifndef ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_
#define ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

BEGIN_ROCPRIM_NAMESPACE




















template<
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_blocked(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread])
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    unsigned int offset = flat_id * ItemsPerThread;
    OutputIterator thread_iter = block_output + offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        thread_iter[item] = items[item];
    }
}


















template<
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_blocked(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread],
                                unsigned int valid)
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    unsigned int offset = flat_id * ItemsPerThread;
    OutputIterator thread_iter = block_output + offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        if (item + offset < valid)
        {
            thread_iter[item] = items[item];
        }
    }
}


























template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_store_direct_blocked_vectorized(unsigned int flat_id,
                                      T* block_output,
                                      U (&items)[ItemsPerThread]) -> typename std::enable_if<detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    static_assert(std::is_convertible<U, T>::value,
                  "The type U must be such that it can be implicitly converted to T.");

    typedef typename detail::match_vector_type<T, ItemsPerThread>::type vector_type;
    constexpr unsigned int vectors_per_thread = (sizeof(T) * ItemsPerThread) / sizeof(vector_type);
    vector_type *vectors_ptr = reinterpret_cast<vector_type*>(const_cast<T*>(block_output));

    vector_type raw_vector_items[vectors_per_thread];
    T *raw_items = reinterpret_cast<T*>(raw_vector_items);

    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        raw_items[item] = items[item];
    }

    block_store_direct_blocked(flat_id, vectors_ptr, raw_vector_items);
}

template<
    class T,
    class U,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto
block_store_direct_blocked_vectorized(unsigned int flat_id,
                                      T* block_output,
                                      U (&items)[ItemsPerThread]) -> typename std::enable_if<!detail::is_vectorizable<T, ItemsPerThread>::value>::type
{
    block_store_direct_blocked(flat_id, block_output, items);
}


















template<
    unsigned int BlockSize,
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_striped(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread])
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    OutputIterator thread_iter = block_output + flat_id;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
         thread_iter[item * BlockSize] = items[item];
    }
}



















template<
    unsigned int BlockSize,
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_striped(unsigned int flat_id,
                                OutputIterator block_output,
                                T (&items)[ItemsPerThread],
                                unsigned int valid)
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    OutputIterator thread_iter = block_output + flat_id;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * BlockSize;
        if (flat_id + offset < valid)
        {
             thread_iter[offset] = items[item];
        }
    }
}

























template<
    unsigned int WarpSize = device_warp_size(),
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_warp_striped(unsigned int flat_id,
                                     OutputIterator block_output,
                                     T (&items)[ItemsPerThread])
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= device_warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;

    OutputIterator thread_iter = block_output + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        thread_iter[item * WarpSize] = items[item];
    }
}


























template<
    unsigned int WarpSize = device_warp_size(),
    class OutputIterator,
    class T,
    unsigned int ItemsPerThread
>
ROCPRIM_DEVICE ROCPRIM_INLINE
void block_store_direct_warp_striped(unsigned int flat_id,
                                     OutputIterator block_output,
                                     T (&items)[ItemsPerThread],
                                     unsigned int valid)
{
    static_assert(std::is_assignable<decltype(block_output[0]), T>::value,
                  "The type T must be such that an object of type OutputIterator "
                  "can be dereferenced and assigned a value of type T.");

    static_assert(detail::is_power_of_two(WarpSize) && WarpSize <= device_warp_size(),
                 "WarpSize must be a power of two and equal or less"
                 "than the size of hardware warp.");
    unsigned int thread_id = detail::logical_lane_id<WarpSize>();
    unsigned int warp_id = flat_id / WarpSize;
    unsigned int warp_offset = warp_id * WarpSize * ItemsPerThread;

    OutputIterator thread_iter = block_output + thread_id + warp_offset;
    ROCPRIM_UNROLL
    for (unsigned int item = 0; item < ItemsPerThread; item++)
    {
        unsigned int offset = item * WarpSize;
        if (warp_offset + thread_id + offset < valid)
        {
            thread_iter[offset] = items[item];
        }
    }
}

END_ROCPRIM_NAMESPACE




#endif
