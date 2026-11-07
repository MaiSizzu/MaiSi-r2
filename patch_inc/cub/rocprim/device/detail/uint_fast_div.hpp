



















#ifndef ROCPRIM_DEVICE_DETAIL_UINT_FAST_DIV_HPP_
#define ROCPRIM_DEVICE_DETAIL_UINT_FAST_DIV_HPP_

#include "../../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

struct uint_fast_div
{
    unsigned int magic;
    unsigned int shift;
    unsigned int add;

    ROCPRIM_HOST_DEVICE inline
    uint_fast_div() = default;

    ROCPRIM_HOST_DEVICE inline
    uint_fast_div(unsigned int d)
    {


        if(d == 1)
        {
            magic = 0;
            shift = 0;
            add = 0;
            return;
        }

        int p;
        unsigned int p32 = 1, q, r, delta;
        add = 0;
        p = 31;
        q = 0x7FFFFFFF/d;
        r = 0x7FFFFFFF - q*d;
        do {
            p = p + 1;
            if(p == 32) p32 = 1;
            else p32 = 2*p32;
            if(r + 1 >= d - r)
            {
                if(q >= 0x7FFFFFFF) add = 1;
                q = 2*q + 1;
                r = 2*r + 1 - d;
            }
            else
            {
                if(q >= 0x80000000) add = 1;
                q = 2*q;
                r = 2*r + 1;
            }
            delta = d - 1 - r;
        } while (p < 64 && p32 < delta);
        magic = q + 1;
        shift = p - 32;

        if(add) shift--;
    }
};

ROCPRIM_HOST_DEVICE inline
unsigned int operator/(unsigned int n, const uint_fast_div& divisor)
{
    if(divisor.magic == 0)
    {

        return n;
    }


    unsigned int q = (static_cast<unsigned long long>(divisor.magic) * static_cast<unsigned long long>(n)) >> 32;
    if(divisor.add)
    {
        q = ((n - q) >> 1) + q;
    }
    return q >> divisor.shift;
}

}

END_ROCPRIM_NAMESPACE

#endif
