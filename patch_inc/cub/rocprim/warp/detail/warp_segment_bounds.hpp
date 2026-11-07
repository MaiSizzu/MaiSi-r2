



















#ifndef ROCPRIM_WARP_DETAIL_WARP_SEGMENT_BOUNDS_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SEGMENT_BOUNDS_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../intrinsics.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{


template<bool HeadSegmented, unsigned int WarpSize, class Flag>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto last_in_warp_segment(Flag flag)
    -> typename std::enable_if<(WarpSize <= __AMDGCN_WAVEFRONT_SIZE), unsigned int>::type
{

    lane_mask_type warp_flags = ::rocprim::ballot(flag);


    if(HeadSegmented)
    {
        warp_flags >>= 1;
    }
    const auto lane_id = ::rocprim::lane_id();

    warp_flags &= lane_mask_type(-1) ^ ((lane_mask_type(1) << lane_id) - 1U);

    warp_flags >>= (lane_id / WarpSize) * WarpSize;

    warp_flags |= lane_mask_type(1) << (WarpSize - 1U);

#ifndef __HIP_CPU_RT__
    #if __AMDGCN_WAVEFRONT_SIZE == 32
    return ::__ffs(warp_flags) - 1;
    #else
    return ::__ffsll(warp_flags) - 1;
    #endif
#else
#if _MSC_VER

    unsigned long tmp = 0;
    _BitScanReverse64(&tmp, warp_flags);
    return 1u << tmp;
#elif __GNUC__
    return __builtin_ctzl(warp_flags);
#else
    static_assert(false, "Look for GCC/Clang implementation");
#endif
#endif
}

}

END_ROCPRIM_NAMESPACE

#endif
