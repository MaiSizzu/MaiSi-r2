



















#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_HPP_

#include <iterator>
#include <iostream>
#include <chrono>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "../functional.hpp"

#include "device_reduce_by_key_config.hpp"
#include "detail/device_reduce_by_key.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class KeyCompareFunction
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void fill_unique_counts_kernel(KeysInputIterator keys_input,
                               unsigned int size,
                               unsigned int * unique_counts,
                               KeyCompareFunction key_compare_op,
                               unsigned int blocks_per_full_batch,
                               unsigned int full_batches)
{
    fill_unique_counts<BlockSize, ItemsPerThread>(
        keys_input, size,
        unique_counts,
        key_compare_op,
        blocks_per_full_batch, full_batches
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class UniqueCountOutputIterator
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void scan_unique_counts_kernel(unsigned int * unique_counts,
                               UniqueCountOutputIterator unique_count_output,
                               unsigned int batches)
{
    scan_unique_counts<BlockSize, ItemsPerThread>(unique_counts, unique_count_output, batches);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Result,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class KeyCompareFunction,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void reduce_by_key_kernel(KeysInputIterator keys_input,
                          ValuesInputIterator values_input,
                          unsigned int size,
                          const unsigned int * unique_starts,
                          carry_out<Result> * carry_outs,
                          Result * leading_aggregates,
                          UniqueOutputIterator unique_output,
                          AggregatesOutputIterator aggregates_output,
                          KeyCompareFunction key_compare_op,
                          BinaryFunction reduce_op,
                          unsigned int blocks_per_full_batch,
                          unsigned int full_batches)
{
    reduce_by_key<BlockSize, ItemsPerThread>(
        keys_input, values_input, size,
        unique_starts, carry_outs, leading_aggregates,
        unique_output, aggregates_output,
        key_compare_op, reduce_op,
        blocks_per_full_batch, full_batches
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Result,
    class AggregatesOutputIterator,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void scan_and_scatter_carry_outs_kernel(const carry_out<Result> * carry_outs,
                                        const Result * leading_aggregates,
                                        AggregatesOutputIterator aggregates_output,
                                        BinaryFunction reduce_op,
                                        unsigned int batches)
{
    scan_and_scatter_carry_outs<BlockSize, ItemsPerThread>(
        carry_outs, leading_aggregates, aggregates_output,
        reduce_op,
        batches
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto _error = cudaGetLastError(); \
        if(_error != cudaSuccess) return _error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto __error = cudaStreamSynchronize(stream); \
            if(__error != cudaSuccess) return __error; \
            auto _end = std::chrono::high_resolution_clock::now(); \
            auto _d = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n'; \
        } \
    }

template<
    class Config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction,
    class KeyCompareFunction
>
inline
cudaError_t reduce_by_key_impl(void * temporary_storage,
                              size_t& storage_size,
                              KeysInputIterator keys_input,
                              ValuesInputIterator values_input,
                              const unsigned int size,
                              UniqueOutputIterator unique_output,
                              AggregatesOutputIterator aggregates_output,
                              UniqueCountOutputIterator unique_count_output,
                              BinaryFunction reduce_op,
                              KeyCompareFunction key_compare_op,
                              const cudaStream_t stream,
                              const bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        typename std::iterator_traits<ValuesInputIterator>::value_type,
        BinaryFunction
    >::type;
    using carry_out_type = carry_out<result_type>;

    using config = default_or_custom_config<
        Config,
        default_reduce_by_key_config<ROCPRIM_TARGET_ARCH, key_type, result_type>
    >;

    constexpr unsigned int items_per_block = config::reduce::block_size * config::reduce::items_per_thread;
    constexpr unsigned int scan_items_per_block = config::scan::block_size * config::scan::items_per_thread;

    const unsigned int blocks = std::max(1u, ::rocprim::detail::ceiling_div(size, items_per_block));
    const unsigned int blocks_per_full_batch = ::rocprim::detail::ceiling_div(blocks, scan_items_per_block);
    const unsigned int full_batches = blocks % scan_items_per_block != 0
        ? blocks % scan_items_per_block
        : scan_items_per_block;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_items_per_block);

    const size_t unique_counts_bytes = ::rocprim::detail::align_size(batches * sizeof(unsigned int));
    const size_t carry_outs_bytes = ::rocprim::detail::align_size(batches * sizeof(carry_out_type));
    const size_t leading_aggregates_bytes = ::rocprim::detail::align_size(batches * sizeof(result_type));
    if(temporary_storage == nullptr)
    {
        storage_size = unique_counts_bytes + carry_outs_bytes + leading_aggregates_bytes;
        return cudaSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "blocks " << blocks << '\n';
        std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
        std::cout << "full_batches " << full_batches << '\n';
        std::cout << "batches " << batches << '\n';
        std::cout << "storage_size " << storage_size << '\n';
        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess) return error;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    unsigned int * unique_counts = reinterpret_cast<unsigned int *>(ptr);
    ptr += unique_counts_bytes;
    carry_out_type * carry_outs = reinterpret_cast<carry_out_type *>(ptr);
    ptr += carry_outs_bytes;
    result_type * leading_aggregates = reinterpret_cast<result_type *>(ptr);


    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    fill_unique_counts_kernel<config::reduce::block_size, config::reduce::items_per_thread>
        <<<dim3(batches), dim3(config::reduce::block_size), 0, stream>>>(
        keys_input, size, unique_counts, key_compare_op,
        blocks_per_full_batch, full_batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("fill_unique_counts", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    scan_unique_counts_kernel<config::scan::block_size, config::scan::items_per_thread>
        <<<dim3(1), dim3(config::scan::block_size), 0, stream>>>(
        unique_counts, unique_count_output,
        batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_unique_counts", config::scan::block_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    reduce_by_key_kernel<config::reduce::block_size, config::reduce::items_per_thread>
        <<<dim3(batches), dim3(config::reduce::block_size), 0, stream>>>(
        keys_input, values_input, size,
        const_cast<const unsigned int *>(unique_counts), carry_outs, leading_aggregates,
        unique_output, aggregates_output,
        key_compare_op, reduce_op,
        blocks_per_full_batch, full_batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("reduce_by_key", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    scan_and_scatter_carry_outs_kernel<config::scan::block_size, config::scan::items_per_thread>
        <<<dim3(1), dim3(config::scan::block_size), 0, stream>>>(
        const_cast<const carry_out_type *>(carry_outs), const_cast<const result_type *>(leading_aggregates),
        aggregates_output,
        reduce_op,
        batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_and_scatter_carry_outs", config::scan::block_size, start)

    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}






































































































template<
    class Config = default_config,
    class KeysInputIterator,
    class ValuesInputIterator,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class UniqueCountOutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
    class KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>
>
inline
cudaError_t reduce_by_key(void * temporary_storage,
                         size_t& storage_size,
                         KeysInputIterator keys_input,
                         ValuesInputIterator values_input,
                         unsigned int size,
                         UniqueOutputIterator unique_output,
                         AggregatesOutputIterator aggregates_output,
                         UniqueCountOutputIterator unique_count_output,
                         BinaryFunction reduce_op = BinaryFunction(),
                         KeyCompareFunction key_compare_op = KeyCompareFunction(),
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false)
{
    return detail::reduce_by_key_impl<Config>(
        temporary_storage, storage_size,
        keys_input, values_input, size,
        unique_output, aggregates_output, unique_count_output,
        reduce_op, key_compare_op,
        stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
