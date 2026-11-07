



















#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_

#include <cmath>
#include <type_traits>
#include <iterator>
#include <iostream>
#include <chrono>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"

#include "device_histogram_config.hpp"
#include "detail/device_histogram.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ActiveChannels,
    class Counter
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void init_histogram_kernel(fixed_array<Counter *, ActiveChannels> histogram,
                           fixed_array<unsigned int, ActiveChannels> bins)
{
    init_histogram<BlockSize, ActiveChannels>(histogram, bins);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void histogram_shared_kernel(SampleIterator samples,
                             unsigned int columns,
                             unsigned int rows,
                             unsigned int row_stride,
                             unsigned int rows_per_block,
                             fixed_array<Counter *, ActiveChannels> histogram,
                             fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                             fixed_array<unsigned int, ActiveChannels> bins)
{
    HIP_DYNAMIC_SHARED(unsigned int, block_histogram);

    histogram_shared<BlockSize, ItemsPerThread, Channels, ActiveChannels>(
        samples, columns, rows, row_stride, rows_per_block,
        histogram,
        sample_to_bin_op, bins,
        block_histogram
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int Channels,
    unsigned int ActiveChannels,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void histogram_global_kernel(SampleIterator samples,
                             unsigned int columns,
                             unsigned int row_stride,
                             fixed_array<Counter *, ActiveChannels> histogram,
                             fixed_array<SampleToBinOp, ActiveChannels> sample_to_bin_op,
                             fixed_array<unsigned int, ActiveChannels> bins_bits)
{
    histogram_global<BlockSize, ItemsPerThread, Channels, ActiveChannels>(
        samples, columns, row_stride,
        histogram,
        sample_to_bin_op, bins_bits
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
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config,
    class SampleIterator,
    class Counter,
    class SampleToBinOp
>
inline
cudaError_t histogram_impl(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int columns,
                          unsigned int rows,
                          size_t row_stride_bytes,
                          Counter * histogram[ActiveChannels],
                          unsigned int levels[ActiveChannels],
                          SampleToBinOp sample_to_bin_op[ActiveChannels],
                          cudaStream_t stream,
                          bool debug_synchronous)
{
    using sample_type = typename std::iterator_traits<SampleIterator>::value_type;

    using config = default_or_custom_config<
        Config,
        default_histogram_config<ROCPRIM_TARGET_ARCH, sample_type, Channels, ActiveChannels>
    >;

    static constexpr unsigned int block_size = config::histogram::block_size;
    static constexpr unsigned int items_per_thread = config::histogram::items_per_thread;
    static constexpr unsigned int items_per_block = block_size * items_per_thread;

    if(row_stride_bytes % sizeof(sample_type) != 0)
    {

        return cudaErrorInvalidValue;
    }

    const unsigned int blocks_x = ::rocprim::detail::ceiling_div(columns, items_per_block);
    const unsigned int row_stride = row_stride_bytes / sizeof(sample_type);

    if(temporary_storage == nullptr)
    {


        storage_size = 4;
        return cudaSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "columns " << columns << '\n';
        std::cout << "rows " << rows << '\n';
        std::cout << "blocks_x " << blocks_x << '\n';
        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess) return error;
    }

    unsigned int bins[ActiveChannels];
    unsigned int bins_bits[ActiveChannels];
    unsigned int total_bins = 0;
    unsigned int max_bins = 0;
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        bins[channel] = levels[channel] - 1;
        bins_bits[channel] = static_cast<unsigned int>(std::log2(detail::next_power_of_two(bins[channel])));
        total_bins += bins[channel];
        max_bins = std::max(max_bins, bins[channel]);
    }

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    init_histogram_kernel<block_size, ActiveChannels><<<dim3(::rocprim::detail::ceiling_div(max_bins, block_size)), dim3(block_size), 0, stream>>>(
        fixed_array<Counter *, ActiveChannels>(histogram),
        fixed_array<unsigned int, ActiveChannels>(bins)
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_histogram", max_bins, start);

    if(columns == 0 || rows == 0)
    {
        return cudaSuccess;
    }

    if(total_bins <= config::shared_impl_max_bins)
    {
        dim3 grid_size;
        grid_size.x = std::min(config::max_grid_size, blocks_x);
        grid_size.y = std::min(rows, config::max_grid_size / grid_size.x);
        const size_t block_histogram_bytes = total_bins * sizeof(unsigned int);
        const unsigned int rows_per_block = ::rocprim::detail::ceiling_div(rows, grid_size.y);
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        histogram_shared_kernel<
                block_size, items_per_thread, Channels, ActiveChannels
            >
            <<<grid_size, dim3(block_size, 1), block_histogram_bytes, stream>>>(
            samples, columns, rows, row_stride, rows_per_block,
            fixed_array<Counter *, ActiveChannels>(histogram),
            fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
            fixed_array<unsigned int, ActiveChannels>(bins)
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_shared", grid_size.x * grid_size.y * block_size, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        histogram_global_kernel<
                block_size, items_per_thread, Channels, ActiveChannels
            >
            <<<dim3(blocks_x, rows), dim3(block_size, 1), 0, stream>>>(
            samples, columns, row_stride,
            fixed_array<Counter *, ActiveChannels>(histogram),
            fixed_array<SampleToBinOp, ActiveChannels>(sample_to_bin_op),
            fixed_array<unsigned int, ActiveChannels>(bins_bits)
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("histogram_global", blocks_x * block_size * rows, start);
    }

    return cudaSuccess;
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t histogram_even_impl(void * temporary_storage,
                               size_t& storage_size,
                               SampleIterator samples,
                               unsigned int columns,
                               unsigned int rows,
                               size_t row_stride_bytes,
                               Counter * histogram[ActiveChannels],
                               unsigned int levels[ActiveChannels],
                               Level lower_level[ActiveChannels],
                               Level upper_level[ActiveChannels],
                               cudaStream_t stream,
                               bool debug_synchronous)
{
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        if(levels[channel] < 2)
        {

            return cudaErrorInvalidValue;
        }
    }

    sample_to_bin_even<Level> sample_to_bin_op[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        sample_to_bin_op[channel] = sample_to_bin_even<Level>(
            levels[channel] - 1,
            lower_level[channel], upper_level[channel]
        );
    }

    return histogram_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, sample_to_bin_op,
        stream, debug_synchronous
    );
}

template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t histogram_range_impl(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int columns,
                                unsigned int rows,
                                size_t row_stride_bytes,
                                Counter * histogram[ActiveChannels],
                                unsigned int levels[ActiveChannels],
                                Level * level_values[ActiveChannels],
                                cudaStream_t stream,
                                bool debug_synchronous)
{
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        if(levels[channel] < 2)
        {

            return cudaErrorInvalidValue;
        }
    }

    sample_to_bin_range<Level> sample_to_bin_op[ActiveChannels];
    for(unsigned int channel = 0; channel < ActiveChannels; channel++)
    {
        sample_to_bin_op[channel] = sample_to_bin_range<Level>(
            levels[channel] - 1,
            level_values[channel]
        );
    }

    return histogram_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, sample_to_bin_op,
        stream, debug_synchronous
    );
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}






































































template<
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t histogram_even(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int size,
                          Counter * histogram,
                          unsigned int levels,
                          Level lower_level,
                          Level upper_level,
                          cudaStream_t stream = 0,
                          bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level lower_level_single[1] = { lower_level };
    Level upper_level_single[1] = { upper_level };

    return detail::histogram_even_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram_single,
        levels_single, lower_level_single, upper_level_single,
        stream, debug_synchronous
    );
}














































































template<
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t histogram_even(void * temporary_storage,
                          size_t& storage_size,
                          SampleIterator samples,
                          unsigned int columns,
                          unsigned int rows,
                          size_t row_stride_bytes,
                          Counter * histogram,
                          unsigned int levels,
                          Level lower_level,
                          Level upper_level,
                          cudaStream_t stream = 0,
                          bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level lower_level_single[1] = { lower_level };
    Level upper_level_single[1] = { upper_level };

    return detail::histogram_even_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram_single,
        levels_single, lower_level_single, upper_level_single,
        stream, debug_synchronous
    );
}















































































template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t multi_histogram_even(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int size,
                                Counter * histogram[ActiveChannels],
                                unsigned int levels[ActiveChannels],
                                Level lower_level[ActiveChannels],
                                Level upper_level[ActiveChannels],
                                cudaStream_t stream = 0,
                                bool debug_synchronous = false)
{
    return detail::histogram_even_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram,
        levels, lower_level, upper_level,
        stream, debug_synchronous
    );
}























































































template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t multi_histogram_even(void * temporary_storage,
                                size_t& storage_size,
                                SampleIterator samples,
                                unsigned int columns,
                                unsigned int rows,
                                size_t row_stride_bytes,
                                Counter * histogram[ActiveChannels],
                                unsigned int levels[ActiveChannels],
                                Level lower_level[ActiveChannels],
                                Level upper_level[ActiveChannels],
                                cudaStream_t stream = 0,
                                bool debug_synchronous = false)
{
    return detail::histogram_even_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, lower_level, upper_level,
        stream, debug_synchronous
    );
}



































































template<
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t histogram_range(void * temporary_storage,
                           size_t& storage_size,
                           SampleIterator samples,
                           unsigned int size,
                           Counter * histogram,
                           unsigned int levels,
                           Level * level_values,
                           cudaStream_t stream = 0,
                           bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level * level_values_single[1] = { level_values };

    return detail::histogram_range_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram_single,
        levels_single, level_values_single,
        stream, debug_synchronous
    );
}











































































template<
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t histogram_range(void * temporary_storage,
                           size_t& storage_size,
                           SampleIterator samples,
                           unsigned int columns,
                           unsigned int rows,
                           size_t row_stride_bytes,
                           Counter * histogram,
                           unsigned int levels,
                           Level * level_values,
                           cudaStream_t stream = 0,
                           bool debug_synchronous = false)
{
    Counter * histogram_single[1] = { histogram };
    unsigned int levels_single[1] = { levels };
    Level * level_values_single[1] = { level_values };

    return detail::histogram_range_impl<1, 1, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram_single,
        levels_single, level_values_single,
        stream, debug_synchronous
    );
}











































































template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t multi_histogram_range(void * temporary_storage,
                                 size_t& storage_size,
                                 SampleIterator samples,
                                 unsigned int size,
                                 Counter * histogram[ActiveChannels],
                                 unsigned int levels[ActiveChannels],
                                 Level * level_values[ActiveChannels],
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    return detail::histogram_range_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, size, 1, 0,
        histogram,
        levels, level_values,
        stream, debug_synchronous
    );
}




















































































template<
    unsigned int Channels,
    unsigned int ActiveChannels,
    class Config = default_config,
    class SampleIterator,
    class Counter,
    class Level
>
inline
cudaError_t multi_histogram_range(void * temporary_storage,
                                 size_t& storage_size,
                                 SampleIterator samples,
                                 unsigned int columns,
                                 unsigned int rows,
                                 size_t row_stride_bytes,
                                 Counter * histogram[ActiveChannels],
                                 unsigned int levels[ActiveChannels],
                                 Level * level_values[ActiveChannels],
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    return detail::histogram_range_impl<Channels, ActiveChannels, Config>(
        temporary_storage, storage_size,
        samples, columns, rows, row_stride_bytes,
        histogram,
        levels, level_values,
        stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
