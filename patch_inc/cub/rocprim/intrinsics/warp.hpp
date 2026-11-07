



















#ifndef ROCPRIM_INTRINSICS_WARP_HPP_
#define ROCPRIM_INTRINSICS_WARP_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE









ROCPRIM_DEVICE ROCPRIM_INLINE
lane_mask_type ballot(int predicate)
{
    return ::__ballot(predicate);
}





ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int masked_bit_count(lane_mask_type x, unsigned int add = 0)
{
    int c;
    #ifndef __HIP_CPU_RT__
        #if __AMDGCN_WAVEFRONT_SIZE == 32
            #ifdef __CUDACC__
            c = ::__builtin_amdgcn_mbcnt_lo(x, add);
            #else
            c = ::__mbcnt_lo(x, add);
            #endif
        #else
            #ifdef __CUDACC__
            c = ::__builtin_amdgcn_mbcnt_lo(static_cast<int>(x), add);
            c = ::__builtin_amdgcn_mbcnt_hi(static_cast<int>(x >> 32), c);
            #else
            c = ::__mbcnt_lo(static_cast<int>(x), add);
            c = ::__mbcnt_hi(static_cast<int>(x >> 32), c);
            #endif
        #endif
    #else
        using namespace hip::detail;
        const auto tidx{id(Fiber::this_fiber()) % warpSize};
        std::bitset<warpSize> bits{x >> (warpSize - tidx)};
        c = static_cast<unsigned int>(bits.count()) + add;
    #endif
    return c;
}

namespace detail
{

ROCPRIM_DEVICE ROCPRIM_INLINE
int warp_any(int predicate)
{
#ifndef __HIP_CPU_RT__
    return ::__any(predicate);
#else
    using namespace hip::detail;
    const auto tidx{id(Fiber::this_fiber()) % warpSize};
    auto& lds{Tile::scratchpad<std::bitset<warpSize>, 1>()[0]};

    lds[tidx] = static_cast<bool>(predicate);

    barrier(Tile::this_tile());

    return lds.any();
#endif
}

ROCPRIM_DEVICE ROCPRIM_INLINE
int warp_all(int predicate)
{
#ifndef __HIP_CPU_RT__
    return ::__all(predicate);
#else
    using namespace hip::detail;
    const auto tidx{id(Fiber::this_fiber()) % warpSize};
    auto& lds{Tile::scratchpad<std::bitset<warpSize>, 1>()[0]};

    lds[tidx] = static_cast<bool>(predicate);

    barrier(Tile::this_tile());

    return lds.all();
#endif
}

}





template <int LABEL_BITS>
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int MatchAny(unsigned int label)
{
    unsigned int retval;


    ROCPRIM_UNROLL
    for (int BIT = 0; BIT < LABEL_BITS; ++BIT)
    {
        unsigned long long  mask;
        unsigned long long current_bit = 1 << BIT;
        mask = label & current_bit;
        bool bit_match = (mask==current_bit);
        mask = ballot(bit_match);
        if(!bit_match)
        {
          mask = ! mask;
        }

        retval = (BIT == 0) ? mask : retval & mask;
    }

    return retval;

}
END_ROCPRIM_NAMESPACE

#endif
