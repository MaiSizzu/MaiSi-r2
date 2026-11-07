



#pragma once

#include <type_traits>

#include "config.hpp"

BEGIN_HIPCUB_NAMESPACE




#ifndef CUB_ALIGN
    #if defined(_WIN32) || defined(_WIN64)

        #define CUB_ALIGN(bytes) __declspec(align(32))
    #else

        #define CUB_ALIGN(bytes) __attribute__((aligned(bytes)))
    #endif
#endif

#define CUB_PREVENT_MACRO_SUBSTITUTION

template <typename T, typename U>
constexpr __host__ __device__ auto min CUB_PREVENT_MACRO_SUBSTITUTION(T &&t,
                                                                      U &&u)
  -> decltype(t < u ? std::forward<T>(t) : std::forward<U>(u))
{
  return t < u ? std::forward<T>(t) : std::forward<U>(u);
}

template <typename T, typename U>
constexpr __host__ __device__ auto max CUB_PREVENT_MACRO_SUBSTITUTION(T &&t,
                                                                      U &&u)
  -> decltype(t < u ? std::forward<U>(u) : std::forward<T>(t))
{
  return t < u ? std::forward<U>(u) : std::forward<T>(t);
}

#ifndef CUB_MAX

    #define CUB_MAX(a, b) (((b) > (a)) ? (b) : (a))
#endif

#ifndef CUB_MIN

    #define CUB_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif

#ifndef CUB_QUOTIENT_FLOOR

    #define CUB_QUOTIENT_FLOOR(x, y) ((x) / (y))
#endif

#ifndef CUB_QUOTIENT_CEILING

    #define CUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))
#endif

#ifndef CUB_ROUND_UP_NEAREST

    #define CUB_ROUND_UP_NEAREST(x, y) ((((x) + (y) - 1) / (y)) * y)
#endif

#ifndef CUB_ROUND_DOWN_NEAREST

    #define CUB_ROUND_DOWN_NEAREST(x, y) (((x) / (y)) * y)
#endif


#ifndef CUB_STATIC_ASSERT
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
        #define CUB_CAT_(a, b) a ## b
        #define CUB_CAT(a, b) CUB_CAT_(a, b)
    #endif


    #define CUB_STATIC_ASSERT(cond, msg) typedef int CUB_CAT(cub_static_assert, __LINE__)[(cond) ? 1 : -1]
#endif



END_HIPCUB_NAMESPACE
