

 #ifndef ROCPRIM_THREAD_THREAD_SCAN_HPP_
 #define ROCPRIM_THREAD_THREAD_SCAN_HPP_

 #include <iterator>
 #include "../config.hpp"

 BEGIN_ROCPRIM_NAMESPACE


template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
ROCPRIM_HOST_DEVICE inline void merge_path_search(
    OffsetT         diagonal,
    AIteratorT      a,
    BIteratorT      b,
    OffsetT         a_len,
    OffsetT         b_len,
    CoordinateT&    path_coordinate)
{

    typedef typename std::iterator_traits<AIteratorT>::value_type T;

    OffsetT split_min = ::rocprim::max(diagonal - b_len, 0);
    OffsetT split_max = ::rocprim::min(diagonal, a_len);

    while (split_min < split_max)
    {
        OffsetT split_pivot = (split_min + split_max) >> 1;
        if (a[split_pivot] <= b[diagonal - split_pivot - 1])
        {

            split_min = split_pivot + 1;
        }
        else
        {

            split_max = split_pivot;
        }
    }

    path_coordinate.x = ::rocprim::min(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}












template <
    typename InputIteratorT,
    typename OffsetT,
    typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT lower_bound(
    InputIteratorT      input,
    OffsetT             num_items,
    T                   val)
{
    OffsetT retval = 0;
    while (num_items > 0)
    {
        OffsetT half = num_items >> 1;
        if (input[retval + half] < val)
        {
            retval = retval + (half + 1);
            num_items = num_items - (half + 1);
        }
        else
        {
            num_items = half;
        }
    }

    return retval;
}










template <
    typename InputIteratorT,
    typename OffsetT,
    typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE OffsetT upper_bound(
    InputIteratorT      input,
    OffsetT             num_items,
    T                   val)
{
    OffsetT retval = 0;
    while (num_items > 0)
    {
        OffsetT half = num_items >> 1;
        if (val < input[retval + half])
        {
            num_items = half;
        }
        else
        {
            retval = retval + (half + 1);
            num_items = num_items - (half + 1);
        }
    }

    return retval;
}

END_ROCPRIM_NAMESPACE

#endif
