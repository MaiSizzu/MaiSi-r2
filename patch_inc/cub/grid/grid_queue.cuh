

#ifndef HIPCUB_ROCPRIM_GRID_GRID_QUEUE_HPP_
#define HIPCUB_ROCPRIM_GRID_GRID_QUEUE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_HIPCUB_NAMESPACE





template <typename OffsetT>
class GridQueue
{
private:


    enum
    {
        FILL    = 0,
        DRAIN   = 1,
    };


    OffsetT *d_counters;

public:


    __host__ __device__ __forceinline__
    static size_t AllocationSize()
    {
        return sizeof(OffsetT) * 2;
    }



    __host__ __device__ __forceinline__ GridQueue()
    :
        d_counters(NULL)
    {}



    __host__ __device__ __forceinline__ GridQueue(
        void *d_storage)
    :
        d_counters((OffsetT*) d_storage)
    {}



    HIPCUB_DEVICE cudaError_t FillAndResetDrain(
        OffsetT fill_size,
        cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        (void)stream;
        d_counters[FILL] = fill_size;
        d_counters[DRAIN] = 0;
        result = cudaSuccess;
        return result;
    }

    HIPCUB_HOST cudaError_t FillAndResetDrain(
        OffsetT fill_size,
        cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        OffsetT counters[2];
        counters[FILL] = fill_size;
        counters[DRAIN] = 0;
        result = CubDebug(cudaMemcpyAsync(d_counters, counters, sizeof(OffsetT) * 2, cudaMemcpyHostToDevice, stream));
        return result;
    }


    HIPCUB_DEVICE cudaError_t ResetDrain(cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        (void)stream;
        d_counters[DRAIN] = 0;
        result = cudaSuccess;
        return result;
    }

    HIPCUB_HOST cudaError_t ResetDrain(cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        result = CubDebug(cudaMemsetAsync(d_counters + DRAIN, 0, sizeof(OffsetT), stream));
        return result;
    }



    HIPCUB_DEVICE cudaError_t ResetFill(cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        (void)stream;
        d_counters[FILL] = 0;
        result = cudaSuccess;
        return result;
    }

    HIPCUB_HOST cudaError_t ResetFill(cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        result = CubDebug(cudaMemsetAsync(d_counters + FILL, 0, sizeof(OffsetT), stream));
        return result;
    }



    HIPCUB_DEVICE cudaError_t FillSize(
        OffsetT &fill_size,
        cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        (void)stream;
        fill_size = d_counters[FILL];
        result = cudaSuccess;
        return result;
    }

    HIPCUB_HOST cudaError_t FillSize(
        OffsetT &fill_size,
        cudaStream_t stream = 0)
    {
        cudaError_t result = cudaErrorUnknown;
        result = CubDebug(cudaMemcpyAsync(&fill_size, d_counters + FILL, sizeof(OffsetT), cudaMemcpyDeviceToHost, stream));
        return result;
    }



    HIPCUB_DEVICE OffsetT Drain(OffsetT num_items)
    {
        return atomicAdd(d_counters + DRAIN, num_items);
    }



    HIPCUB_DEVICE OffsetT Fill(OffsetT num_items)
    {
        return atomicAdd(d_counters + FILL, num_items);
    }
};


#ifndef DOXYGEN_SHOULD_SKIP_THIS



template <typename OffsetT>
__global__ void FillAndResetDrainKernel(
    GridQueue<OffsetT>   grid_queue,
    OffsetT              num_items)
{
    grid_queue.FillAndResetDrain(num_items);
}



#endif




END_HIPCUB_NAMESPACE

#endif
