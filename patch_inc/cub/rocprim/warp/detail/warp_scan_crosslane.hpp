



















#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_CROSSLANE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_CROSSLANE_HPP_

#include <type_traits>

#include "../../config.hpp"

#include "warp_scan_dpp.hpp"
#include "warp_scan_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    bool UseDPP = ROCPRIM_DETAIL_USE_DPP
>
using warp_scan_crosslane =
    typename std::conditional<
        UseDPP,
        warp_scan_dpp<T, WarpSize>,
        warp_scan_shuffle<T, WarpSize>
    >::type;

}

END_ROCPRIM_NAMESPACE

#endif
