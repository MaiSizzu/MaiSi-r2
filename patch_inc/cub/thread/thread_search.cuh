

 #ifndef HIBCUB_ROCPRIM_THREAD_THREAD_SEARCH_HPP_
 #define HIBCUB_ROCPRIM_THREAD_THREAD_SEARCH_HPP_

 #include <iterator>

 BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS


template <
    typename AIteratorT,
    typename BIteratorT,
    typename OffsetT,
    typename CoordinateT>
__host__ __device__ __forceinline__ void MergePathSearch(
    OffsetT         diagonal,
    AIteratorT      a,
    BIteratorT      b,
    OffsetT         a_len,
    OffsetT         b_len,
    CoordinateT&    path_coordinate)
{
    OffsetT split_min = CUB_MAX(diagonal - b_len, 0);
    OffsetT split_max = CUB_MIN(diagonal, a_len);

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

    path_coordinate.x = CUB_MIN(split_min, a_len);
    path_coordinate.y = diagonal - split_min;
}




template <
    typename InputIteratorT,
    typename OffsetT,
    typename T>
__device__ __forceinline__ OffsetT LowerBound(
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
__device__ __forceinline__ OffsetT UpperBound(
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

#endif

END_HIPCUB_NAMESPACE

#endif
