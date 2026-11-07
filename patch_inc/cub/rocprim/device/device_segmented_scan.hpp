



















#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "../iterator/zip_iterator.hpp"
#include "../iterator/discard_iterator.hpp"
#include "../iterator/transform_iterator.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../types/tuple.hpp"

#include "device_scan_config.hpp"
#include "device_scan.hpp"
#include "detail/device_segmented_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    bool Exclusive,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void segmented_scan_kernel(InputIterator input,
                           OutputIterator output,
                           OffsetIterator begin_offsets,
                           OffsetIterator end_offsets,
                           InitValueType initial_value,
                           BinaryFunction scan_op)
{
    segmented_scan<Exclusive, Config, ResultType>(
        input, output, begin_offsets, end_offsets,
        static_cast<ResultType>(initial_value), scan_op
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
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
inline
cudaError_t segmented_scan_impl(void * temporary_storage,
                               size_t& storage_size,
                               InputIterator input,
                               OutputIterator output,
                               unsigned int segments,
                               OffsetIterator begin_offsets,
                               OffsetIterator end_offsets,
                               const InitValueType initial_value,
                               BinaryFunction scan_op,
                               cudaStream_t stream,
                               bool debug_synchronous)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename std::conditional<Exclusive, InitValueType, input_type>::type;


    using config = default_or_custom_config<
        Config,
        default_scan_config<ROCPRIM_TARGET_ARCH, result_type>
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
    segmented_scan_kernel<Exclusive, config, result_type>
        <<<dim3(segments), dim3(block_size), 0, stream>>>(
        input, output,
        begin_offsets, end_offsets,
        initial_value, scan_op
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_scan", segments, start);
    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}























































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t segmented_inclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    unsigned int segments,
                                    OffsetIterator begin_offsets,
                                    OffsetIterator end_offsets,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    cudaStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = input_type;

    return detail::segmented_scan_impl<false, Config>(
        temporary_storage, storage_size,
        input, output, segments, begin_offsets, end_offsets, result_type(),
        scan_op, stream, debug_synchronous
    );
}




























































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t segmented_exclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    unsigned int segments,
                                    OffsetIterator begin_offsets,
                                    OffsetIterator end_offsets,
                                    const InitValueType initial_value,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    cudaStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    return detail::segmented_scan_impl<true, Config>(
        temporary_storage, storage_size,
        input, output, segments, begin_offsets, end_offsets, initial_value,
        scan_op, stream, debug_synchronous
    );
}















































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class HeadFlagIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t segmented_inclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    HeadFlagIterator head_flags,
                                    size_t size,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    cudaStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = input_type;
    using flag_type = typename std::iterator_traits<HeadFlagIterator>::value_type;
    using headflag_scan_op_wrapper_type =
        detail::headflag_scan_op_wrapper<
            result_type, flag_type, BinaryFunction
        >;

    return inclusive_scan<Config>(
        temporary_storage, storage_size,
        rocprim::make_zip_iterator(rocprim::make_tuple(input, head_flags)),
        rocprim::make_zip_iterator(rocprim::make_tuple(output, rocprim::make_discard_iterator())),
        size, headflag_scan_op_wrapper_type(scan_op),
        stream, debug_synchronous
    );
}


















































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class HeadFlagIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t segmented_exclusive_scan(void * temporary_storage,
                                    size_t& storage_size,
                                    InputIterator input,
                                    OutputIterator output,
                                    HeadFlagIterator head_flags,
                                    const InitValueType initial_value,
                                    size_t size,
                                    BinaryFunction scan_op = BinaryFunction(),
                                    cudaStream_t stream = 0,
                                    bool debug_synchronous = false)
{
    using result_type = InitValueType;
    using flag_type = typename std::iterator_traits<HeadFlagIterator>::value_type;
    using headflag_scan_op_wrapper_type =
        detail::headflag_scan_op_wrapper<
            result_type, flag_type, BinaryFunction
        >;

    const result_type initial_value_converted = static_cast<result_type>(initial_value);



    return exclusive_scan<Config>(
        temporary_storage, storage_size,
        rocprim::make_transform_iterator(
            rocprim::make_counting_iterator<size_t>(0),
            [input, head_flags, initial_value_converted, size]
            ROCPRIM_DEVICE
            (const size_t i)
            {
                flag_type flag(false);
                if(i + 1 < size)
                {
                    flag = head_flags[i + 1];
                }
                result_type value = initial_value_converted;
                if(!flag)
                {
                    value = input[i];
                }
                return rocprim::make_tuple(value, flag);
            }
        ),
        rocprim::make_zip_iterator(rocprim::make_tuple(output, rocprim::make_discard_iterator())),
        rocprim::make_tuple(initial_value_converted, flag_type(true)),
        size,
        headflag_scan_op_wrapper_type(scan_op),
        stream,
        debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
