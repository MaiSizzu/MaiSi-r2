

#ifndef ROCPRIM_THREAD_THREAD_REDUCE_HPP_
#define ROCPRIM_THREAD_THREAD_REDUCE_HPP_


#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE










template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp,
    bool        NoPrefix = false>
ROCPRIM_DEVICE ROCPRIM_INLINE T thread_reduce(
    T*           input,
    ReductionOp reduction_op,
    T           prefix = T(0))
{
    T retval;
    if(NoPrefix)
        retval = input[0];
    else
        retval = prefix;

    ROCPRIM_UNROLL
    for (int i = 0 + NoPrefix; i < LENGTH; ++i)
        retval = reduction_op(retval, input[i]);

    return retval;
}









template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
ROCPRIM_DEVICE ROCPRIM_INLINE T thread_reduce(
    T           (&input)[LENGTH],
    ReductionOp reduction_op,
    T           prefix)
{
    return thread_reduce<LENGTH, false>((T*)input, reduction_op, prefix);
}








template <
    int         LENGTH,
    typename    T,
    typename    ReductionOp>
ROCPRIM_DEVICE ROCPRIM_INLINE T thread_reduce(
    T           (&input)[LENGTH],
    ReductionOp reduction_op)
{
    return thread_reduce<LENGTH, true>((T*)input, reduction_op);
}

END_ROCPRIM_NAMESPACE

#endif
