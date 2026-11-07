

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_

#include <limits>
#include <iterator>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "../config.hpp"
#include "../iterator/arg_index_input_iterator.cuh"
#include "../thread/thread_operators.cuh"

#include <cub/rocprim/device/device_reduce.hpp>
#include <cub/rocprim/device/device_reduce_by_key.hpp>

BEGIN_HIPCUB_NAMESPACE
namespace detail
{

template<class T>
inline
T get_lowest_value()
{
    return std::numeric_limits<T>::lowest();
}

template<>
inline
__half get_lowest_value<__half>()
{
    unsigned short lowest_half = 0xfbff;
    __half lowest_value = *reinterpret_cast<__half*>(&lowest_half);
    return lowest_value;
}

template<>
inline
__nv_bfloat16 get_lowest_value<__nv_bfloat16>()
{
    return __float2bfloat16(-3.38953138925e+38f);
}

template<class T>
inline
T get_max_value()
{
    return std::numeric_limits<T>::max();
}

template<>
inline
__half get_max_value<__half>()
{
    unsigned short max_half = 0x7bff;
    __half max_value = *reinterpret_cast<__half*>(&max_half);
    return max_value;
}

template<>
inline
__nv_bfloat16 get_max_value<__nv_bfloat16>()
{
    return __float2bfloat16(3.38953138925e+38f);
}

}

class DeviceReduce
{
public:
    template <
        typename InputIteratorT,
        typename OutputIteratorT,
        typename ReduceOpT,
        typename T
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Reduce(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      ReduceOpT reduction_op,
                      T init,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        return (cudaError_t)::rocprim::reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, init, num_items,
            cub::detail::convert_result_type<InputIteratorT, OutputIteratorT>(reduction_op),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Sum(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   cudaStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, cub::Sum(), T(0),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Min(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   cudaStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, cub::Min(), detail::get_max_value<T>(),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ArgMin(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT =
            typename std::conditional<
                std::is_same<O, void>::value,
                KeyValuePair<OffsetT, T>,
                O
            >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        OutputTupleT init(1, detail::get_max_value<T>());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out, num_items, cub::ArgMin(), init,
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t Max(void *d_temp_storage,
                   size_t &temp_storage_bytes,
                   InputIteratorT d_in,
                   OutputIteratorT d_out,
                   int num_items,
                   cudaStream_t stream = 0,
                   bool debug_synchronous = false)
    {
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_in, d_out, num_items, cub::Max(), detail::get_lowest_value<T>(),
            stream, debug_synchronous
        );
    }

    template <
        typename InputIteratorT,
        typename OutputIteratorT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ArgMax(void *d_temp_storage,
                      size_t &temp_storage_bytes,
                      InputIteratorT d_in,
                      OutputIteratorT d_out,
                      int num_items,
                      cudaStream_t stream = 0,
                      bool debug_synchronous = false)
    {
        using OffsetT = int;
        using T = typename std::iterator_traits<InputIteratorT>::value_type;
        using O = typename std::iterator_traits<OutputIteratorT>::value_type;
        using OutputTupleT =
            typename std::conditional<
                std::is_same<O, void>::value,
                KeyValuePair<OffsetT, T>,
                O
            >::type;

        using OutputValueT = typename OutputTupleT::Value;
        using IteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

        IteratorT d_indexed_in(d_in);
        OutputTupleT init(1, detail::get_lowest_value<T>());

        return Reduce(
            d_temp_storage, temp_storage_bytes,
            d_indexed_in, d_out, num_items, cub::ArgMax(), init,
            stream, debug_synchronous
        );
    }

    template<
        typename KeysInputIteratorT,
        typename UniqueOutputIteratorT,
        typename ValuesInputIteratorT,
        typename AggregatesOutputIteratorT,
        typename NumRunsOutputIteratorT,
        typename ReductionOpT
    >
    HIPCUB_RUNTIME_FUNCTION static
    cudaError_t ReduceByKey(void * d_temp_storage,
                           size_t& temp_storage_bytes,
                           KeysInputIteratorT d_keys_in,
                           UniqueOutputIteratorT d_unique_out,
                           ValuesInputIteratorT d_values_in,
                           AggregatesOutputIteratorT d_aggregates_out,
                           NumRunsOutputIteratorT d_num_runs_out,
                           ReductionOpT reduction_op,
                           int num_items,
                           cudaStream_t stream = 0,
                           bool debug_synchronous = false)
    {
        using key_compare_op =
            ::rocprim::equal_to<typename std::iterator_traits<KeysInputIteratorT>::value_type>;
        return (cudaError_t)::rocprim::reduce_by_key(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_values_in, num_items,
            d_unique_out, d_aggregates_out, d_num_runs_out,
            cub::detail::convert_result_type<ValuesInputIteratorT, AggregatesOutputIteratorT>(reduction_op),
            key_compare_op(),
            stream, debug_synchronous
        );
    }
};

END_HIPCUB_NAMESPACE

#endif
