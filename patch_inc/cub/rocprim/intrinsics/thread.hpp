



















#ifndef ROCPRIM_INTRINSICS_THREAD_HPP_
#define ROCPRIM_INTRINSICS_THREAD_HPP_

#include <atomic>

#include "../config.hpp"
#include "../detail/various.hpp"

BEGIN_ROCPRIM_NAMESPACE











ROCPRIM_HOST_DEVICE inline
constexpr unsigned int warp_size()
{
    return warpSize;
}





ROCPRIM_HOST inline
unsigned int host_warp_size()
{
    int default_hip_device;
    cudaError_t success = cudaGetDevice(&default_hip_device);
    cudaDeviceProp device_prop;
    success = cudaGetDeviceProperties(&device_prop,default_hip_device);

    if(success != cudaSuccess)
        return -1;
    else
        return device_prop.warpSize;
};





ROCPRIM_DEVICE ROCPRIM_INLINE
constexpr unsigned int device_warp_size()
{
    return warpSize;
}


ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_block_size()
{
    return blockDim.z * blockDim.y * blockDim.x;
}


ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_tile_size()
{
    return flat_block_size();
}




ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int lane_id()
{
#ifndef __HIP_CPU_RT__
    return ::__lane_id();
#else
    using namespace hip::detail;
    return id(Fiber::this_fiber()) % warpSize;
#endif
}


ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_block_thread_id()
{
    return (threadIdx.z * blockDim.y * blockDim.x)
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
}


template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_thread_id()
    -> typename std::enable_if<(BlockSizeY == 1 && BlockSizeZ == 1), unsigned int>::type
{
    return threadIdx.x;
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_thread_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ == 1), unsigned int>::type
{
    return threadIdx.x + (threadIdx.y * blockDim.x);
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_thread_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ > 1), unsigned int>::type
{
    return threadIdx.x + (threadIdx.y * blockDim.x) +
           (threadIdx.z * blockDim.y * blockDim.x);
}


ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_tile_thread_id()
{
    return flat_block_thread_id();
}


ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int warp_id()
{
    return flat_block_thread_id()/device_warp_size();
}

ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int warp_id(unsigned int flat_id)
{
    return flat_id/device_warp_size();
}


template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int warp_id()
{
    return flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>()/device_warp_size();
}


ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int flat_block_id()
{
    return (blockIdx.z * gridDim.y * gridDim.x)
        + (blockIdx.y * gridDim.x)
        + blockIdx.x;
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_id()
    -> typename std::enable_if<(BlockSizeY == 1 && BlockSizeZ == 1), unsigned int>::type
{
    return blockIdx.x;
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ == 1), unsigned int>::type
{
    return blockIdx.x + (blockIdx.y * gridDim.x);
}

template<unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto flat_block_id()
    -> typename std::enable_if<(BlockSizeY > 1 && BlockSizeZ > 1), unsigned int>::type
{
    return blockIdx.x + (blockIdx.y * gridDim.x) +
           (blockIdx.z * gridDim.y * gridDim.x);
}




ROCPRIM_DEVICE ROCPRIM_INLINE
void syncthreads()
{
    __syncthreads();
}



ROCPRIM_DEVICE ROCPRIM_INLINE
void wave_barrier()
{
    __builtin_amdgcn_wave_barrier();
}

namespace detail
{

    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int block_thread_id()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");

        return 0;
    }


    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int block_id()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");

        return 0;
    }


    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int block_size()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");

        return 0;
    }


    template<unsigned int Dim>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int grid_size()
    {
        static_assert(Dim > 2, "Dim must be 0, 1 or 2");

        return 0;
    }

    #define ROCPRIM_DETAIL_CONCAT(A, B) A B
    #define ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, dim, suffix) \
        template<> \
        ROCPRIM_DEVICE ROCPRIM_INLINE \
        unsigned int name<dim>() \
        { \
            return ROCPRIM_DETAIL_CONCAT(prefix, suffix); \
        }
    #define ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(name, prefix) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 0, x) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 1, y) \
        ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC(name, prefix, 2, z)

    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_thread_id, threadIdx.)
    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_id, blockIdx.)
    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(block_size, blockDim.)
    ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS(grid_size, gridDim.)

    #undef ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNCS
    #undef ROCPRIM_DETAIL_DEFINE_HIP_API_ID_FUNC
    #undef ROCPRIM_DETAIL_CONCAT


    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto logical_lane_id()
        -> typename std::enable_if<detail::is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return lane_id() & (LogicalWarpSize-1);
    }

    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto logical_lane_id()
        -> typename std::enable_if<!detail::is_power_of_two(LogicalWarpSize), unsigned int>::type
    {
        return lane_id()%LogicalWarpSize;
    }

    template<>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int logical_lane_id<device_warp_size()>()
    {
        return lane_id();
    }


    template<unsigned int LogicalWarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int logical_warp_id()
    {
        return flat_block_thread_id()/LogicalWarpSize;
    }

    template<>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int logical_warp_id<device_warp_size()>()
    {
        return warp_id();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void memory_fence_system()
    {
        ::__threadfence_system();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void memory_fence_block()
    {
        ::__threadfence_block();
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void memory_fence_device()
    {
        ::__threadfence();
    }
}




END_ROCPRIM_NAMESPACE

#endif
