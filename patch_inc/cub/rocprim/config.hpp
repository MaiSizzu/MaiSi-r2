



















#ifndef ROCPRIM_CONFIG_HPP_
#define ROCPRIM_CONFIG_HPP_

#define BEGIN_ROCPRIM_NAMESPACE \
    namespace rocprim {

#define END_ROCPRIM_NAMESPACE \
    } 

#include <limits>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef ROCPRIM_DEVICE
    #define ROCPRIM_DEVICE __device__
    #define ROCPRIM_HOST __host__
    #define ROCPRIM_HOST_DEVICE __host__ __device__
    #define ROCPRIM_SHARED_MEMORY __shared__
    #ifdef WIN32
    #define ROCPRIM_KERNEL __global__ static
    #else
    #define ROCPRIM_KERNEL __global__
    #endif

    #ifndef ROCPRIM_DEFAULT_MAX_BLOCK_SIZE
        #define ROCPRIM_DEFAULT_MAX_BLOCK_SIZE 256
    #endif
    #ifndef ROCPRIM_DEFAULT_MIN_WARPS_PER_EU
        #define ROCPRIM_DEFAULT_MIN_WARPS_PER_EU 1
    #endif



    #ifdef WIN32
      #define ROCPRIM_INLINE inline __attribute__((always_inline))
    #else
      #define ROCPRIM_INLINE inline
    #endif
    #define ROCPRIM_FORCE_INLINE __attribute__((always_inline))
#endif

#ifndef ROCPRIM_DISABLE_DPP
    #define ROCPRIM_DETAIL_USE_DPP true
#endif

#ifdef ROCPRIM_DISABLE_LOOKBACK_SCAN
    #define ROCPRIM_DETAIL_USE_LOOKBACK_SCAN false
#else
    #define ROCPRIM_DETAIL_USE_LOOKBACK_SCAN true
#endif

#ifndef ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS
    #define ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS 1
#endif

#ifndef ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS
    #define ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS 1
#endif








#ifndef ROCPRIM_TARGET_ARCH
    #define ROCPRIM_TARGET_ARCH 0
#endif

#if (__gfx1010__ || __gfx1011__ || __gfx1012__ || __gfx1030__ || __gfx1031__)
    #define ROCPRIM_NAVI 1
#else
    #define ROCPRIM_NAVI 0
#endif
#define ROCPRIM_ARCH_90a 910


#define ROCPRIM_WARP_SIZE_32 32u
#define ROCPRIM_WARP_SIZE_64 64u
#define ROCPRIM_MAX_WARP_SIZE ROCPRIM_WARP_SIZE_64

#if (defined(_MSC_VER) && !defined(__clang__)) || (defined(__GNUC__) && !defined(__clang__))
#define ROCPRIM_UNROLL
#define ROCPRIM_NO_UNROLL
#else
#define ROCPRIM_UNROLL _Pragma("unroll")
#define ROCPRIM_NO_UNROLL _Pragma("nounroll")
#endif

#ifndef ROCPRIM_GRID_SIZE_LIMIT
#define ROCPRIM_GRID_SIZE_LIMIT std::numeric_limits<unsigned int>::max()
#endif

#if __cpp_if_constexpr >= 201606
#define ROCPRIM_IF_CONSTEXPR constexpr
#else
#define ROCPRIM_IF_CONSTEXPR
#endif

#endif
