

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_

#include <limits>
#include <iterator>

#include "../config.hpp"

#include "../thread/thread_operators.cuh"
#include "../iterator/arg_index_input_iterator.cuh"

#include <cub/rocprim/device/device_segmented_reduce.hpp>

BEGIN_HIPCUB_NAMESPACE

struct DeviceSegmentedReduce
{
    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT,
        typename ReductionOp,
        typename T
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Reduce(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_segments,
                      OffsetIteratorT d_begin_offsets,
                      OffsetIteratorT d_end_offsets,
                      ReductionOp reduction_op,
                      T initial_value,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::segmented_reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            cub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
            initial_value,
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Sum(void * d_temp_storage,
                   size_t& temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_segments,
                   OffsetIteratorT d_begin_offsets,
                   OffsetIteratorT d_end_offsets,
                   cudaStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            cub::Sum(), input_type(),
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Min(void * d_temp_storage,
                   size_t& temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_segments,
                   OffsetIteratorT d_begin_offsets,
                   OffsetIteratorT d_end_offsets,
                   cudaStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            cub::Min(), std::numeric_limits<input_type>::max(),
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ArgMin(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_segments,
                      OffsetIteratorT d_begin_offsets,
                      OffsetIteratorT d_end_offsets,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT = typename std::conditional<
                                 std::is_same<O, void>::value,
                                 KeyValuePair<OffsetT, T>,
                                 O
                             >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        const OutputTupleT init(1, std::numeric_limits<T>::max());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            cub::ArgMin(), init,
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Max(void * d_temp_storage,
                   size_t& temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_segments,
                   OffsetIteratorT d_begin_offsets,
                   OffsetIteratorT d_end_offsets,
                   cudaStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using input_type = typename std::iterator_traits<InputIteratorT>::value_type;

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            cub::Max(), std::numeric_limits<input_type>::lowest(),
            stream, debug_synchronous
        );
    }

    template<
        typename InputIteratorT,
        typename OutputIteratorT,
        typename OffsetIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ArgMax(void * d_temp_storage,
                      size_t& temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_segments,
                      OffsetIteratorT d_begin_offsets,
                      OffsetIteratorT d_end_offsets,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT = typename std::conditional<
                                 std::is_same<O, void>::value,
                                 KeyValuePair<OffsetT, T>,
                                 O
                             >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        const OutputTupleT init(1, std::numeric_limits<T>::lowest());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out,
            num_segments, d_begin_offsets, d_end_offsets,
            cub::ArgMax(), init,
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
