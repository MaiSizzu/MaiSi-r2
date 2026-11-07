



















#ifndef ROCPRIM_INTRINSICS_ATOMIC_HPP_
#define ROCPRIM_INTRINSICS_ATOMIC_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_add(unsigned int * address, unsigned int value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    int atomic_add(int * address, int value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    float atomic_add(float * address, float value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_add(unsigned long long * address, unsigned long long value)
    {
        return ::atomicAdd(address, value);
    }
    

    ROCPRIM_DEVICE ROCPRIM_INLINE
    long long atomic_add(long long * address, long long value)
    {
        return ::atomicAdd((unsigned long long*)address, (unsigned long long)value);
    }


    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_wrapinc(unsigned int * address, unsigned int value)
    {
        return ::atomicInc(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_exch(unsigned int * address, unsigned int value)
    {
        return ::atomicExch(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_exch(unsigned long long * address, unsigned long long value)
    {
        return ::atomicExch(address, value);
    }
}

END_ROCPRIM_NAMESPACE

#endif
