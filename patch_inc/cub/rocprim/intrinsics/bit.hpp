



















#ifndef ROCPRIM_INTRINSICS_BIT_HPP_
#define ROCPRIM_INTRINSICS_BIT_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE





ROCPRIM_DEVICE ROCPRIM_INLINE
int get_bit(int x, int i)
{
    return (x >> i) & 1;
}




ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int bit_count(unsigned int x)
{
    return __popc(x);
}




ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int bit_count(unsigned long long x)
{
    return __popcll(x);
}




END_ROCPRIM_NAMESPACE

#endif
