

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SPMV_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SPMV_HPP_

#include "../config.hpp"

#include "../iterator/tex_ref_input_iterator.cuh"



#define DTK_24_04_PRE
BEGIN_HIPCUB_NAMESPACE


class DeviceSpmv
{

public:

template <
    typename        ValueT,
    typename        OffsetT>
struct SpmvParams
{
    ValueT*         d_values;
    OffsetT*        d_row_end_offsets;
    OffsetT*        d_column_indices;
    ValueT*         d_vector_x;
    ValueT*         d_vector_y;
    int             num_rows;
    int             num_cols;
    int             num_nonzeros;
    ValueT          alpha;
    ValueT          beta;

    cub::TexRefInputIterator<ValueT, 66778899, OffsetT>  t_vector_x;
};

static constexpr uint32_t CsrMVKernel_MaxThreads = 256;

template <typename ValueT>
static __global__ void
CsrMVKernel(SpmvParams<ValueT, int> spmv_params)
{
    __shared__ ValueT partial;

    const int32_t row_id = blockIdx.x;

    if(threadIdx.x == 0)
    {
        partial = spmv_params.beta * spmv_params.d_vector_y[row_id];
    }
    __syncthreads();

    int32_t row_offset = (row_id == 0) ? (0) : (spmv_params.d_row_end_offsets[row_id - 1]);
    for(uint32_t thread_offset = 0; thread_offset < spmv_params.num_cols / blockDim.x; thread_offset++)
    {
        int32_t offset = row_offset + thread_offset * blockDim.x + threadIdx.x;

        if(offset < spmv_params.d_row_end_offsets[row_id])
        {
            ValueT t_value =
                spmv_params.alpha *
                spmv_params.d_values[offset] *
                spmv_params.d_vector_x[spmv_params.d_column_indices[offset]];

            atomicAdd(&partial, t_value);

            __syncthreads();

            if(threadIdx.x == 0)
            {
                spmv_params.d_vector_y[row_id] = partial;
            }
        }
    }
}


template <typename ValueT>
static __global__ __launch_bounds__(CsrMVKernel_MaxThreads) void
CsrMVKernelV2(SpmvParams<ValueT, int> spmv_params)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t warp_id = thread_id / 64;
    const int32_t lane = thread_id % 64;

    const int32_t row = warp_id; 

    ValueT dot = 0;
    if (row < spmv_params.num_rows)
    {
        const int32_t row_start = spmv_params.d_row_end_offsets[row - 1];
        const int32_t row_end =  spmv_params.d_row_end_offsets[row];

        for (int32_t element = row_start + lane; element < row_end; element += 64)
        {
            dot +=  spmv_params.alpha *
                    spmv_params.d_values[element] *
                    spmv_params.d_vector_x[spmv_params.d_column_indices[element]];
        }

    }

    for(unsigned int s= 32; s>0; s >>= 1) 
    {
        dot += __shfl_down(dot, s);
    }
    if (lane == 0 && row < spmv_params.num_rows)
    {
        spmv_params.d_vector_y[row] = dot + spmv_params.beta * spmv_params.d_vector_y[row];
    }
}
static HIPCUB_DEVICE unsigned int prev_power_of_2 (unsigned int n)
{
    while (n & n - 1)
        n = n & n - 1;
    return n;
}


template <typename ValueT>
    HIPCUB_RUNTIME_FUNCTION
    static cudaError_t CsrMV(
        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        ValueT*             d_values,
        int*                d_row_offsets,
        int*                d_column_indices,
        ValueT*             d_vector_x,
        ValueT*             d_vector_y,
        int                 num_rows,
        int                 num_cols,
        int                 num_nonzeros,
        cudaStream_t         stream                  = 0,
        bool                debug_synchronous       = false)
    {
        SpmvParams<ValueT, int> spmv_params;
        spmv_params.d_values             = d_values;
        spmv_params.d_row_end_offsets    = d_row_offsets + 1;
        spmv_params.d_column_indices     = d_column_indices;
        spmv_params.d_vector_x           = d_vector_x;
        spmv_params.d_vector_y           = d_vector_y;
        spmv_params.num_rows             = num_rows;
        spmv_params.num_cols             = num_cols;
        spmv_params.num_nonzeros         = num_nonzeros;
        spmv_params.alpha                = 1.0;
        spmv_params.beta                 = 0.0;

        cudaError_t status;
        if(d_temp_storage == nullptr)
        {


            temp_storage_bytes = 4;
            return cudaError_t(0);
        }
        else
        {

            #ifndef DTK_24_04_PRE
            size_t block_size = min(num_cols, DeviceSpmv::CsrMVKernel_MaxThreads);
            size_t grid_size = num_rows;
            CsrMVKernel<<<grid_size, block_size, 0, stream>>>(spmv_params);
            status = cudaGetLastError();
            #else
            size_t block_size = 256;
            const unsigned int warp_size = 64;
            size_t grid_size; 
            grid_size = ((size_t)warp_size * (size_t)num_rows + block_size - 1) / block_size;
            CsrMVKernelV2<<<grid_size, block_size, 0, stream>>>(spmv_params);
            status = cudaGetLastError();
            #endif

        }
        return status;
    }


template <typename ValueT>
static __global__ __launch_bounds__(CsrMVKernel_MaxThreads) void
CsrAdaptiveKernel(SpmvParams<ValueT, int> spmv_params,int* d_row_block)
{
    const size_t row_start = d_row_block[blockIdx.x];
    const size_t row_end = d_row_block[blockIdx.x + 1];
    const size_t nnz = spmv_params.d_row_end_offsets[row_end] - spmv_params.d_row_end_offsets[row_start];
    const int nnz_per_wg = blockDim.x;


    __shared__ ValueT cache[1024];

    if(row_end - row_start > 1)
    {
        const unsigned int i = threadIdx.x;
        const unsigned int block_data_begin = spmv_params.d_row_end_offsets[row_start];
        const unsigned int thread_data_begin = block_data_begin + i;
        if(i < nnz)
        {
            cache[i] =  spmv_params.alpha *
                        spmv_params.d_values[thread_data_begin] *
                        spmv_params.d_vector_x[spmv_params.d_column_indices[thread_data_begin]];
        }
        __syncthreads ();
 
        const unsigned int threads_for_reduction = prev_power_of_2 (blockDim.x / (row_end - row_start));
       
        if (threads_for_reduction > 1)
        {

            const unsigned int thread_in_block = i % threads_for_reduction;
            const unsigned int local_row = row_start + i / threads_for_reduction;

            ValueT dot = 0.0;
            
            if (local_row < row_end)
            {
                const unsigned int local_first_element = spmv_params.d_row_end_offsets[local_row] - spmv_params.d_row_end_offsets[row_start];
                const unsigned int local_last_element = spmv_params.d_row_end_offsets[local_row + 1] - spmv_params.d_row_end_offsets[row_start];
                for (unsigned int local_element = local_first_element + thread_in_block;
                    local_element < local_last_element;
                    local_element += threads_for_reduction)
                {
                 
                    dot += cache[local_element];
                }
            }
            __syncthreads ();
            cache[i] = dot;
            for (int j = threads_for_reduction / 2; j > 0; j /= 2)
            {

                __syncthreads ();

                const bool use_result = thread_in_block < j && i + j < nnz_per_wg;

                if (use_result)
                    dot += cache[i + j];
                __syncthreads ();

                if (use_result)
                    cache[i] = dot;
            }

            if (thread_in_block == 0 && local_row < row_end)
                spmv_params.d_vector_y[local_row] = dot + spmv_params.beta * spmv_params.d_vector_y[local_row];
        }
        else
        {
            unsigned int local_row = row_start + i;
            while (local_row < row_end)
            {
                ValueT dot = 0.0;

                for (unsigned int j = spmv_params.d_row_end_offsets[local_row] - block_data_begin;
                    j < spmv_params.d_row_end_offsets[local_row + 1] - block_data_begin;
                    j++)
                {
                    dot += cache[j];
                }

                spmv_params.d_vector_y[local_row] = dot + spmv_params.beta * spmv_params.d_vector_y[local_row];
                local_row += nnz_per_wg;
            }
        }

    }
    else
    {
        const unsigned int row = row_start;
        const unsigned int warp_id = threadIdx.x / 64;
        const unsigned int lane = threadIdx.x % 64;
        ValueT sum = 0;

        if(nnz <= blockDim.x || blockDim.x <= 64)
        {
            if(row < spmv_params.num_rows)
            {
                const unsigned int nnz_start = spmv_params.d_row_end_offsets[row];
                const unsigned int nnz_end = spmv_params.d_row_end_offsets[row + 1];
                for(unsigned int element = nnz_start + lane; element < nnz_end; element += 64)
                {
                    sum += spmv_params.alpha *
                        spmv_params.d_values[element] *
                        spmv_params.d_vector_x[spmv_params.d_column_indices[element]];
                }
            }
            for(unsigned int s= 32; s>0; s >>= 1) 
            {
                sum += __shfl_down(sum, s);
            }
            if (lane == 0 && warp_id == 0 && row < spmv_params.num_rows)
            {
                spmv_params.d_vector_y[row] = sum + spmv_params.beta * spmv_params.d_vector_y[row];
            }
        }
        else
        {
            if (row < spmv_params.num_rows)
            {
                const unsigned int nnz_start = spmv_params.d_row_end_offsets[row];
                const unsigned int nnz_end = spmv_params.d_row_end_offsets[row + 1];

                for (unsigned int element = nnz_start + threadIdx.x; element < nnz_end; element += blockDim.x)
                {
                    sum += spmv_params.alpha *
                        spmv_params.d_values[element] *
                        spmv_params.d_vector_x[spmv_params.d_column_indices[element]];
                }

            }
            for(unsigned int s= 32; s>0; s >>= 1) 
            {
                sum += __shfl_down(sum, s);
            }

            if (lane == 0)
                cache[warp_id] = sum;
            __syncthreads ();

            if (warp_id == 0)
            {
                sum = 0;
                for (unsigned int element = lane; element < blockDim.x / 64; element += 64)
                    sum += cache[element];

                for(unsigned int s = 32; s>0; s >>= 1) 
                {
                    sum += __shfl_down(sum, s);
                }
                if (lane == 0 && row < spmv_params.num_rows)
                {
                    spmv_params.d_vector_y[row] = sum + spmv_params.beta * spmv_params.d_vector_y[row];
                }
            }

        }
        
    }

}

template <typename ValueT>
    HIPCUB_RUNTIME_FUNCTION
    static cudaError_t csrAdaptivyMV(
        void*               d_temp_storage,
        size_t&             temp_storage_bytes,
        ValueT*             d_values,
        int*                d_row_offsets,
        int*                d_column_indices,
        ValueT*             d_vector_x,
        ValueT*             d_vector_y,
        int                 num_rows,
        int                 num_cols,
        int                 num_nonzeros,
        int*                d_row_block,
        int&                d_block_number, 
        int&                d_block_size,                      
        cudaStream_t         stream                  = 0,
        bool                debug_synchronous       = false)
    {
        SpmvParams<ValueT, int> spmv_params;
        spmv_params.d_values             = d_values;
        spmv_params.d_row_end_offsets    = d_row_offsets;
        spmv_params.d_column_indices     = d_column_indices;
        spmv_params.d_vector_x           = d_vector_x;
        spmv_params.d_vector_y           = d_vector_y;
        spmv_params.num_rows             = num_rows;
        spmv_params.num_cols             = num_cols;
        spmv_params.num_nonzeros         = num_nonzeros;
        spmv_params.alpha                = 1.0;
        spmv_params.beta                 = 0.0;
        
        cudaError_t status;
        if(d_temp_storage == nullptr)
        {


            temp_storage_bytes = 4;
            return cudaError_t(0);
        }
        else
        {
            size_t grid_size = d_block_number;
            size_t block_size = d_block_size;
            CsrAdaptiveKernel<<<grid_size, block_size, 0, stream>>>(spmv_params,d_row_block);
            status = cudaGetLastError();
        }
        return status;
    }
};


END_HIPCUB_NAMESPACE

#endif

