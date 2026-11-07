



















#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_BINARY_SEARCH_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_BINARY_SEARCH_HPP_

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Size>
ROCPRIM_DEVICE ROCPRIM_INLINE
Size get_binary_search_middle(Size left, Size right)
{
    const Size d = right - left;
    return left + d / 2 + d / 64;
}

template<class RandomAccessIterator, class Size, class T, class BinaryPredicate>
ROCPRIM_DEVICE ROCPRIM_INLINE
Size lower_bound_n(RandomAccessIterator first,
                   Size size,
                   const T& value,
                   BinaryPredicate compare_op)
{
    Size left = 0;
    Size right = size;
    while(left < right)
    {
        const Size mid = get_binary_search_middle(left, right);
        if(compare_op(first[mid], value))
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
    return left;
}

template<class RandomAccessIterator, class Size, class T, class BinaryPredicate>
ROCPRIM_DEVICE ROCPRIM_INLINE
Size upper_bound_n(RandomAccessIterator first,
                   Size size,
                   const T& value,
                   BinaryPredicate compare_op)
{
    Size left = 0;
    Size right = size;
    while(left < right)
    {
        const Size mid = get_binary_search_middle(left, right);
        if(compare_op(value, first[mid]))
        {
            right = mid;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

struct lower_bound_search_op
{
    template<class HaystackIterator, class CompareOp, class Size, class T>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    Size operator()(HaystackIterator haystack, Size size, const T& value, CompareOp compare_op) const
    {
        return lower_bound_n(haystack, size, value, compare_op);
    }
};

struct upper_bound_search_op
{
    template<class HaystackIterator, class CompareOp, class Size, class T>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    Size operator()(HaystackIterator haystack, Size size, const T& value, CompareOp compare_op) const
    {
        return upper_bound_n(haystack, size, value, compare_op);
    }
};

struct binary_search_op
{
    template<class HaystackIterator, class CompareOp, class Size, class T>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(HaystackIterator haystack, Size size, const T& value, CompareOp compare_op) const
    {
        const Size n = lower_bound_n(haystack, size, value, compare_op);
        return n != size && !compare_op(value, haystack[n]);
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
