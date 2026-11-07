

#ifndef HIPCUB_ROCPRIM_ITERATOR_CACHE_MODIFIED_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_CACHE_MODIFIED_INPUT_ITERATOR_HPP_

#include <iterator>
#include <iostream>

#include "../thread/thread_load.cuh"
#include "../util_type.cuh"

#if (THRUST_VERSION >= 100700)

    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif

BEGIN_HIPCUB_NAMESPACE

template <
    CacheLoadModifier   MODIFIER,
    typename            ValueType,
    typename            OffsetT = ptrdiff_t>
class CacheModifiedInputIterator
{
public:


    typedef CacheModifiedInputIterator          self_type;
    typedef OffsetT                             difference_type;
    typedef ValueType                           value_type;
    typedef ValueType*                          pointer;
    typedef ValueType                           reference;
    typedef std::random_access_iterator_tag     iterator_category;

public:


    ValueType* ptr;


    __host__ __device__ __forceinline__ CacheModifiedInputIterator(
        ValueType* ptr)
    :
        ptr(const_cast<typename std::remove_cv<ValueType>::type *>(ptr))
    {}


    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        ptr++;
        return retval;
    }


    __host__ __device__ __forceinline__ self_type operator++()
    {
        ptr++;
        return *this;
    }


    __device__ __forceinline__ reference operator*() const
    {
        return ThreadLoad<MODIFIER>(ptr);
    }


    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(ptr + n);
        return retval;
    }


    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        ptr += n;
        return *this;
    }


    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(ptr - n);
        return retval;
    }


    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        ptr -= n;
        return *this;
    }


    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return ptr - other.ptr;
    }


    template <typename Distance>
    __device__ __forceinline__ reference operator[](Distance n) const
    {
        return ThreadLoad<MODIFIER>(ptr + n);
    }


    __device__ __forceinline__ pointer operator->()
    {
        return &ThreadLoad<MODIFIER>(ptr);
    }


    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }


    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS


    friend std::ostream& operator<<(std::ostream& os, const self_type& )
    {
        return os;
    }

#endif

};

END_HIPCUB_NAMESPACE

#endif
