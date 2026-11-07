



















#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_REDUCE_HPP_

#include <type_traits>
#include <iterator>
#include <chrono>

#include "device_reduce_config.hpp"

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "detail/device_segmented_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    class Config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void segmented_reduce_kernel(InputIterator input,
                             OutputIterator output,
                             OffsetIterator begin_offsets,
                             OffsetIterator end_offsets,
                             BinaryFunction reduce_op,
                             ResultType initial_value)
{
    segmented_reduce<Config>(
        input, output,
        begin_offsets, end_offsets,
        reduce_op, initial_value
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
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
inline
cudaError_t segmented_reduce_impl(void * temporary_storage,
                                 size_t& storage_size,
                                 InputIterator input,
                                 OutputIterator output,
                                 unsigned int segments,
                                 OffsetIterator begin_offsets,
                                 OffsetIterator end_offsets,
                                 BinaryFunction reduce_op,
                                 InitValueType initial_value,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::detail::match_result_type<
        input_type, BinaryFunction
    >::type;


    using config = default_or_custom_config<
        Config,
        default_reduce_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    constexpr unsigned int block_size = config::block_size;

    if(temporary_storage == nullptr)
    {


        storage_size = 4;
        return cudaSuccess;
    }

    if( segments == 0u )
        return cudaSuccess;

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    segmented_reduce_kernel<config>
        <<<dim3(segments), dim3(block_size), 0, stream>>>(
        input, output,
        begin_offsets, end_offsets,
        reduce_op, static_cast<result_type>(initial_value)
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_reduce", segments, start);

    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}































































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>,
    class InitValueType = typename std::iterator_traits<InputIterator>::value_type
>
inline
cudaError_t segmented_reduce(void * temporary_storage,
                            size_t& storage_size,
                            InputIterator input,
                            OutputIterator output,
                            unsigned int segments,
                            OffsetIterator begin_offsets,
                            OffsetIterator end_offsets,
                            BinaryFunction reduce_op = BinaryFunction(),
                            InitValueType initial_value = InitValueType(),
                            cudaStream_t stream = 0,
                            bool debug_synchronous = false)
{
    return detail::segmented_reduce_impl<Config>(
        temporary_storage, storage_size,
        input, output,
        segments, begin_offsets, end_offsets,
        reduce_op, initial_value,
        stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
