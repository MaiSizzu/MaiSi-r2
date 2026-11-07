

#ifndef HIPCUB_ROCPRIM_ITERATOR_TEX_REF_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_TEX_REF_INPUT_ITERATOR_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"

#if (THRUST_VERSION >= 100700)
    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif

#include <rocprim/iterator/texture_cache_iterator.hpp>

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int UNIQUE_ID,
    typename OffsetT = std::ptrdiff_t
>
class TexRefInputIterator : public ::rocprim::texture_cache_iterator<T, OffsetT>
{
    public:
    template<class Qualified>
    inline
    cudaError_t BindTexture(Qualified* ptr,
                           size_t bytes = size_t(-1),
                           size_t texture_offset = 0)
    {
        return (cudaError_t)::rocprim::texture_cache_iterator<T, OffsetT>::bind_texture(ptr, bytes, texture_offset);
    }

    inline cudaError_t UnbindTexture()
    {
        return (cudaError_t)::rocprim::texture_cache_iterator<T, OffsetT>::unbind_texture();
    }

    HIPCUB_HOST_DEVICE inline
    ~TexRefInputIterator() = default;

    HIPCUB_HOST_DEVICE inline
    TexRefInputIterator() : ::rocprim::texture_cache_iterator<T, OffsetT>()
    {
    }

    HIPCUB_HOST_DEVICE inline
    TexRefInputIterator(const ::rocprim::texture_cache_iterator<T, OffsetT> other)
        : ::rocprim::texture_cache_iterator<T, OffsetT>(other)
    {
    }

};

END_HIPCUB_NAMESPACE

#endif
