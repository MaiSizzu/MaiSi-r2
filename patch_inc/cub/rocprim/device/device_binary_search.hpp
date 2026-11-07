



















#ifndef ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_
#define ROCPRIM_DEVICE_DEVICE_BINARY_SEARCH_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "detail/device_binary_search.hpp"

#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    class Config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class SearchFunction,
    class CompareFunction
>
inline
cudaError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         SearchFunction search_op,
                         CompareFunction compare_op,
                         cudaStream_t stream,
                         bool debug_synchronous)
{
    using value_type = typename std::iterator_traits<NeedlesIterator>::value_type;

    if(temporary_storage == nullptr)
    {


        storage_size = 4;
        return cudaSuccess;
    }

    return transform<Config>(
        needles, output,
        needles_size,
        [haystack, haystack_size, search_op, compare_op]
        ROCPRIM_DEVICE
        (const value_type& value)
        {
            return search_op(haystack, haystack_size, value, compare_op);
        },
        stream, debug_synchronous
    );
}

}

template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
cudaError_t lower_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       cudaStream_t stream = 0,
                       bool debug_synchronous = false)
{
    return detail::binary_search<Config>(
        temporary_storage, storage_size,
        haystack, needles, output,
        haystack_size, needles_size,
        detail::lower_bound_search_op(), compare_op,
        stream, debug_synchronous
    );
}

template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
cudaError_t upper_bound(void * temporary_storage,
                       size_t& storage_size,
                       HaystackIterator haystack,
                       NeedlesIterator needles,
                       OutputIterator output,
                       size_t haystack_size,
                       size_t needles_size,
                       CompareFunction compare_op = CompareFunction(),
                       cudaStream_t stream = 0,
                       bool debug_synchronous = false)
{
    return detail::binary_search<Config>(
        temporary_storage, storage_size,
        haystack, needles, output,
        haystack_size, needles_size,
        detail::upper_bound_search_op(), compare_op,
        stream, debug_synchronous
    );
}

template<
    class Config = default_config,
    class HaystackIterator,
    class NeedlesIterator,
    class OutputIterator,
    class CompareFunction = ::rocprim::less<>
>
inline
cudaError_t binary_search(void * temporary_storage,
                         size_t& storage_size,
                         HaystackIterator haystack,
                         NeedlesIterator needles,
                         OutputIterator output,
                         size_t haystack_size,
                         size_t needles_size,
                         CompareFunction compare_op = CompareFunction(),
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false)
{
    return detail::binary_search<Config>(
        temporary_storage, storage_size,
        haystack, needles, output,
        haystack_size, needles_size,
        detail::binary_search_op(), compare_op,
        stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
