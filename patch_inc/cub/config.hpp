

#ifndef HIPCUB_CONFIG_HPP_
#define HIPCUB_CONFIG_HPP_

#include <cuda_runtime.h>
#include "util_namespace.cuh"



#define BEGIN_HIPCUB_NAMESPACE CUB_NAMESPACE_BEGIN
#define END_HIPCUB_NAMESPACE CUB_NAMESPACE_END


#ifndef HIPCUB_ARCH
#define HIPCUB_ARCH 1
#endif

#define CUB_DEVICE_WARP_THREADS 64

#ifdef __CUDACC__
    #define HIPCUB_ROCPRIM_API 1
    #define HIPCUB_RUNTIME_FUNCTION __host__
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #define HIPCUB_CUB_API 1
    #define HIPCUB_RUNTIME_FUNCTION CUB_RUNTIME_FUNCTION

    #include <cub/util_arch.cuh>
    #define HIPCUB_WARP_THREADS CUB_PTX_WARP_THREADS
    #define HIPCUB_DEVICE_WARP_THREADS CUB_PTX_WARP_THREADS
    #define HIPCUB_HOST_WARP_THREADS CUB_PTX_WARP_THREADS
    #define HIPCUB_ARCH CUB_PTX_ARCH
    BEGIN_HIPCUB_NAMESPACE
    using namespace cub;
    END_HIPCUB_NAMESPACE
#endif


#define HIPCUB_WARP_SIZE_32 32u
#define HIPCUB_WARP_SIZE_64 64u
#define HIPCUB_MAX_WARP_SIZE HIPCUB_WARP_SIZE_64

#define HIPCUB_HOST __host__
#define HIPCUB_DEVICE __device__
#define HIPCUB_HOST_DEVICE __host__ __device__
#define HIPCUB_SHARED_MEMORY __shared__


#ifdef __clang__
#define HIPCUB_PRAGMA_TO_STR(x) _Pragma(#x)
#define HIPCUB_CLANG_SUPPRESS_WARNING_PUSH _Pragma("clang diagnostic push")
#define HIPCUB_CLANG_SUPPRESS_WARNING(w) HIPCUB_PRAGMA_TO_STR(clang diagnostic ignored w)
#define HIPCUB_CLANG_SUPPRESS_WARNING_POP _Pragma("clang diagnostic pop")
#define HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH(w) \
    HIPCUB_CLANG_SUPPRESS_WARNING_PUSH HIPCUB_CLANG_SUPPRESS_WARNING(w)
#else
#define HIPCUB_CLANG_SUPPRESS_WARNING_PUSH
#define HIPCUB_CLANG_SUPPRESS_WARNING(w)
#define HIPCUB_CLANG_SUPPRESS_WARNING_POP
#define HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)
#endif

BEGIN_HIPCUB_NAMESPACE


#if (defined(DEBUG) || defined(_DEBUG)) && !defined(HIPCUB_STDERR)
    #define HIPCUB_STDERR
#endif

inline
cudaError_t Debug(
    cudaError_t      error,
    const char*     filename,
    int             line)
{
    (void)filename;
    (void)line;
#ifdef HIPCUB_STDERR
    if (error)
    {
        fprintf(stderr, "cuda error %d [%s, %d]: %s\n", error, filename, line, cudaGetErrorString(error));
        fflush(stderr);
    }
#endif
    return error;
}
#ifndef cubDebug
    #define cubDebug(e) cub::Debug((cudaError_t) (e), __FILE__, __LINE__)
#endif


#ifndef CubDebug
    #define CubDebug(e) cub::Debug((cudaError_t) (e), __FILE__, __LINE__)
#endif

#ifndef CubDebugExit
    #define CubDebugExit(e) if (cub::Debug((cudaError_t) (e), __FILE__, __LINE__)) { exit(1); }
#endif

END_HIPCUB_NAMESPACE

#endif
