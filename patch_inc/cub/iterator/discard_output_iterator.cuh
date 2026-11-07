

#ifndef HIPCUB_ROCPRIM_ITERATOR_DISCARD_OUTPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_DISCARD_OUTPUT_ITERATOR_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"

BEGIN_HIPCUB_NAMESPACE
#if (THRUST_VERSION >= 100700)

    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif





template <typename OffsetT = ptrdiff_t>
class DiscardOutputIterator
{
public:


    typedef DiscardOutputIterator   self_type;
    typedef OffsetT                 difference_type;
    typedef void                    value_type;
    typedef void                    pointer;
    typedef void                    reference;

#if (THRUST_VERSION >= 100700)

    typedef typename thrust::detail::iterator_facade_category<
        thrust::any_system_tag,
        thrust::random_access_traversal_tag,
        value_type,
        reference
      >::type iterator_category;
#else
    typedef std::random_access_iterator_tag     iterator_category;
#endif

private:

    OffsetT offset;

public:


    __host__ __device__ __forceinline__ DiscardOutputIterator(
        OffsetT offset = 0)
    :
        offset(offset)
    {}

    
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        offset++;
        return retval;
    }

    
    __host__ __device__ __forceinline__ self_type operator++()
    {
        offset++;
        return *this;
    }

    
    __host__ __device__ __forceinline__ self_type& operator*()
    {

        return *this;
    }

    
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(offset + n);
        return retval;
    }

    
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        offset += n;
        return *this;
    }

    
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(offset - n);
        return retval;
    }

    
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        offset -= n;
        return *this;
    }

    
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return offset - other.offset;
    }

    
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator[](Distance)
    {

        return *this;
    }


    __host__ __device__ __forceinline__ pointer operator->()
    {
        return;
    }


    template<typename T>
    __host__ __device__ __forceinline__ void operator=(T const&)
    {}


    __host__ __device__ __forceinline__ operator void*() const { return NULL; }

    
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (offset == rhs.offset);
    }

    
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (offset != rhs.offset);
    }

    
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        os << "[" << itr.offset << "]";
        return os;
    }
};

END_HIPCUB_NAMESPACE

#endif
