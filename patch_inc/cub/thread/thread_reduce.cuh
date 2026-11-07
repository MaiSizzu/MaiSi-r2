

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_REDUCE_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_REDUCE_HPP_

BEGIN_HIPCUB_NAMESPACE


namespace internal {

template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp,
    bool        NoPrefix = false>
__device__ __forceinline__ T ThreadReduce(
    T*           input,
    ReductionOp reduction_op,
    T           prefix = T(0))
{
    T retval;
    if(NoPrefix)
        retval = input[0];
    else
        retval = prefix;

    #pragma unroll
    for (int i = 0 + NoPrefix; i < LENGTH; ++i)
        retval = reduction_op(retval, input[i]);

    return retval;
}

template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T           (&input)[LENGTH],
    ReductionOp reduction_op,
    T           prefix)
{
    return ThreadReduce<LENGTH, false>((T*)input, reduction_op, prefix);
}

template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
__device__ __forceinline__ T ThreadReduce(
    T           (&input)[LENGTH],
    ReductionOp reduction_op)
{
    return ThreadReduce<LENGTH, true>((T*)input, reduction_op);
}

}

END_HIPCUB_NAMESPACE

#endif
