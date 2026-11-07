

#ifndef HIPCUB_ROCPRIM_GRID_GRID_BARRIER_HPP_
#define HIPCUB_ROCPRIM_GRID_GRID_BARRIER_HPP_

#include <type_traits>

#include "../config.hpp"

#include "../thread/thread_load.cuh"

BEGIN_HIPCUB_NAMESPACE





class GridBarrier
{
protected :

    typedef unsigned int SyncFlag;


    SyncFlag* d_sync;

public:

    
    GridBarrier() : d_sync(NULL) {}

     
    __device__ __forceinline__ void Sync() const
    {
        volatile SyncFlag *d_vol_sync = d_sync;



        __threadfence();
        __syncthreads();

        if (blockIdx.x == 0)
        {

            if (threadIdx.x == 0)
            {
                d_vol_sync[blockIdx.x] = 1;
            }

            __syncthreads();


            for (uint32_t peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x)
            {
                while (ThreadLoad<LOAD_CG>(d_sync + peer_block) == 0)
                {
                    __threadfence_block();
                }
            }

            __syncthreads();


            for (uint32_t peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x)
            {
                d_vol_sync[peer_block] = 0;
            }
        }
        else
        {
            if (threadIdx.x == 0)
            {

                d_vol_sync[blockIdx.x] = 1;


                while (ThreadLoad<LOAD_CG>(d_sync + blockIdx.x) == 1)
                {
                    __threadfence_block();
                }
            }

            __syncthreads();
        }
    }
};



class GridBarrierLifetime : public GridBarrier
{
protected:


    size_t sync_bytes;

public:

    
    GridBarrierLifetime() : GridBarrier(), sync_bytes(0) {}


    
    cudaError_t HostReset()
    {
        cudaError_t retval = cudaSuccess;
        if (d_sync)
        {
            retval = cudaFree(d_sync);
            d_sync = NULL;
        }
        sync_bytes = 0;
        return retval;
    }


    
    virtual ~GridBarrierLifetime()
    {
        HostReset();
    }


    
    cudaError_t Setup(int sweep_grid_size)
    {
        cudaError_t retval = cudaSuccess;
        do {
            size_t new_sync_bytes = sweep_grid_size * sizeof(SyncFlag);
            if (new_sync_bytes > sync_bytes)
            {
                if (d_sync)
                {
                    if ((retval = cudaFree(d_sync))) break;
                }

                sync_bytes = new_sync_bytes;


                if ((retval = cudaMalloc((void**) &d_sync, sync_bytes))) break;
                if ((retval = cudaMemset(d_sync, 0, new_sync_bytes))) break;
            }
        } while (0);

        return retval;
    }
};

END_HIPCUB_NAMESPACE

#endif
