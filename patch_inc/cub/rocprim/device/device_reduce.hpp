



















#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_HPP_

#include <type_traits>
#include <iterator>
#include <algorithm>
#include <chrono>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"

#include "device_reduce_config.hpp"
#include "detail/device_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    bool WithInitialValue,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void block_reduce_kernel(InputIterator input,
                         const size_t size,
                         OutputIterator output,
                         InitValueType initial_value,
                         BinaryFunction reduce_op)
{
    block_reduce_kernel_impl<WithInitialValue, Config, ResultType>(
        input, size, output, initial_value, reduce_op
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC(name, size, start) \
    if(debug_synchronous) \
    { \
        std::cout << name << "(" << size << ")"; \
        auto _error = cudaStreamSynchronize(stream); \
        if(_error != cudaSuccess) return _error; \
        auto _end = std::chrono::high_resolution_clock::now(); \
        auto _d = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
        std::cout << " " << _d.count() * 1000 << " ms" << '\n'; \
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
    bool WithInitialValue,
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
cudaError_t reduce_impl(void * temporary_storage,
                       size_t& storage_size,
                       InputIterator input,
                       OutputIterator output,
                       const InitValueType initial_value,
                       const size_t size,
                       BinaryFunction reduce_op,
                       const cudaStream_t stream,
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
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    if(temporary_storage == nullptr)
    {
        storage_size = reduce_get_temporary_storage_bytes<result_type>(size, items_per_block);

        storage_size = storage_size == 0 ? 4 : storage_size;
        return cudaSuccess;
    }


    std::chrono::high_resolution_clock::time_point start;

    static constexpr auto size_limit             = config::size_limit;
    static constexpr auto number_of_blocks_limit = ::rocprim::max<size_t>(size_limit / items_per_block, 1);

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "number of blocks limit " << number_of_blocks_limit << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    if(number_of_blocks > 1)
    {

        result_type * block_prefixes = static_cast<result_type*>(temporary_storage);
        static constexpr auto aligned_size_limit = number_of_blocks_limit * items_per_block;


        const auto number_of_launch = (size + aligned_size_limit - 1) / aligned_size_limit;
        for(size_t i = 0, offset = 0; i < number_of_launch; ++i, offset += aligned_size_limit) {
            const auto current_size = std::min<size_t>(size - offset, aligned_size_limit);
            const auto current_blocks = (current_size + items_per_block - 1) / items_per_block;

            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            detail::block_reduce_kernel<false, config, result_type>
                <<<dim3(current_blocks),
                dim3(block_size),
                0,
                stream>>>(
                input + offset,
                current_size,
                block_prefixes + i * number_of_blocks_limit,
                initial_value,
                reduce_op);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", current_size, start);
        }

        void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
        auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(result_type));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        auto error = reduce_impl<WithInitialValue, config>(
            nested_temp_storage,
            nested_temp_storage_size,
            block_prefixes,
            output,
            initial_value,
            number_of_blocks,
            reduce_op,
            stream,
            debug_synchronous
        );
        if(error != cudaSuccess) return error;
        ROCPRIM_DETAIL_HIP_SYNC("nested_device_reduce", number_of_blocks, start);
    }
    else
    {
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        detail::block_reduce_kernel<WithInitialValue, config, result_type>
            <<<dim3(1), dim3(block_size), 0, stream>>>(
            input, size, output, initial_value, reduce_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", size, start);
    }

    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

}




























































































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t reduce(void * temporary_storage,
                 size_t& storage_size,
                 InputIterator input,
                 OutputIterator output,
                 const InitValueType initial_value,
                 const size_t size,
                 BinaryFunction reduce_op = BinaryFunction(),
                 const cudaStream_t stream = 0,
                 bool debug_synchronous = false)
{
    return detail::reduce_impl<true, Config>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        reduce_op, stream, debug_synchronous
    );
}











































































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t reduce(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  const size_t size,
                  BinaryFunction reduce_op = BinaryFunction(),
                  const cudaStream_t stream = 0,
                  bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;

    return detail::reduce_impl<false, Config>(
        temporary_storage, storage_size,
        input, output, input_type(), size,
        reduce_op, stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
