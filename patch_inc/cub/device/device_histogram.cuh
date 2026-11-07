

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_HISTOGRAM_HPP_

#include "../config.hpp"

#include "../util_type.cuh"

#include <cub/rocprim/device/device_histogram.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DeviceHistogram
{
    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t HistogramEven(void * d_temp_storage,
                             size_t& temp_storage_bytes,
                             SampleIteratorT d_samples,
                             CounterT * d_histogram,
                             int num_levels,
                             LevelT lower_level,
                             LevelT upper_level,
                             OffsetT num_samples,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::histogram_even(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_samples,
            d_histogram,
            num_levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t HistogramEven(void * d_temp_storage,
                             size_t& temp_storage_bytes,
                             SampleIteratorT d_samples,
                             CounterT * d_histogram,
                             int num_levels,
                             LevelT lower_level,
                             LevelT upper_level,
                             OffsetT num_row_samples,
                             OffsetT num_rows,
                             size_t row_stride_bytes,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::histogram_even(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_samples, num_rows, row_stride_bytes,
            d_histogram,
            num_levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t MultiHistogramEven(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  SampleIteratorT d_samples,
                                  CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                  int num_levels[NUM_ACTIVE_CHANNELS],
                                  LevelT lower_level[NUM_ACTIVE_CHANNELS],
                                  LevelT upper_level[NUM_ACTIVE_CHANNELS],
                                  OffsetT num_pixels,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return (cudaError_t)::rocprim::multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_pixels,
            d_histogram,
            levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t MultiHistogramEven(void * d_temp_storage,
                                  size_t& temp_storage_bytes,
                                  SampleIteratorT d_samples,
                                  CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                  int num_levels[NUM_ACTIVE_CHANNELS],
                                  LevelT lower_level[NUM_ACTIVE_CHANNELS],
                                  LevelT upper_level[NUM_ACTIVE_CHANNELS],
                                  OffsetT num_row_pixels,
                                  OffsetT num_rows,
                                  size_t row_stride_bytes,
                                  cudaStream_t stream = 0,
                                  bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return (cudaError_t)::rocprim::multi_histogram_even<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_pixels, num_rows, row_stride_bytes,
            d_histogram,
            levels, lower_level, upper_level,
            stream, debug_synchronous
        );
    }

    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t HistogramRange(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              SampleIteratorT d_samples,
                              CounterT * d_histogram,
                              int num_levels,
                              LevelT * d_levels,
                              OffsetT num_samples,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::histogram_range(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_samples,
            d_histogram,
            num_levels, d_levels,
            stream, debug_synchronous
        );
    }

    template<
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t HistogramRange(void * d_temp_storage,
                              size_t& temp_storage_bytes,
                              SampleIteratorT d_samples,
                              CounterT * d_histogram,
                              int num_levels,
                              LevelT * d_levels,
                              OffsetT num_row_samples,
                              OffsetT num_rows,
                              size_t row_stride_bytes,
                              cudaStream_t stream = 0,
                              bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::histogram_range(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_samples, num_rows, row_stride_bytes,
            d_histogram,
            num_levels, d_levels,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t MultiHistogramRange(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   SampleIteratorT d_samples,
                                   CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                   int num_levels[NUM_ACTIVE_CHANNELS],
                                   LevelT * d_levels[NUM_ACTIVE_CHANNELS],
                                   OffsetT num_pixels,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return (cudaError_t)::rocprim::multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_pixels,
            d_histogram,
            levels, d_levels,
            stream, debug_synchronous
        );
    }

    template<
        int NUM_CHANNELS,
        int NUM_ACTIVE_CHANNELS,
        typename SampleIteratorT,
        typename CounterT,
        typename LevelT,
        typename OffsetT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t MultiHistogramRange(void * d_temp_storage,
                                   size_t& temp_storage_bytes,
                                   SampleIteratorT d_samples,
                                   CounterT * d_histogram[NUM_ACTIVE_CHANNELS],
                                   int num_levels[NUM_ACTIVE_CHANNELS],
                                   LevelT * d_levels[NUM_ACTIVE_CHANNELS],
                                   OffsetT num_row_pixels,
                                   OffsetT num_rows,
                                   size_t row_stride_bytes,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false)
    {
        unsigned int levels[NUM_ACTIVE_CHANNELS];
        for(unsigned int channel = 0; channel < NUM_ACTIVE_CHANNELS; channel++)
        {
            levels[channel] = num_levels[channel];
        }
        return (cudaError_t)::rocprim::multi_histogram_range<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
            d_temp_storage, temp_storage_bytes,
            d_samples, num_row_pixels, num_rows, row_stride_bytes,
            d_histogram,
            levels, d_levels,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
