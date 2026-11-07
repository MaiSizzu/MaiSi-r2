



















#ifndef ROCPRIM_WARP_DETAIL_WARP_REDUCE_CROSSLANE_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_REDUCE_CROSSLANE_HPP_

#include <type_traits>

#include "../../config.hpp"

#include "warp_reduce_dpp.hpp"
#include "warp_reduce_shuffle.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize,
    bool UseAllReduce,
    bool UseDPP = ROCPRIM_DETAIL_USE_DPP
>
using warp_reduce_crosslane =
    typename std::conditional<
        UseDPP,
        warp_reduce_dpp<T, WarpSize, UseAllReduce>,
        warp_reduce_shuffle<T, WarpSize, UseAllReduce>
    >::type;

}

END_ROCPRIM_NAMESPACE

#endif
