



















#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_HPP_

#include <algorithm>
#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/match_result_type.hpp"
#include "../types/tuple.hpp"
#include "../iterator/zip_iterator.hpp"

#include "device_transform_config.hpp"
#include "detail/device_transform.hpp"
#include <chrono>

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class UnaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void transform_kernel(InputIterator input,
                      const size_t size,
                      OutputIterator output,
                      UnaryFunction transform_op)
{
    transform_kernel_impl<BlockSize, ItemsPerThread, ResultType>(
        input, size, output, transform_op
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        auto _error = cudaGetLastError(); \
        if(_error != cudaSuccess) return _error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            _error = cudaStreamSynchronize(stream); \
            if(_error != cudaSuccess) return _error; \
            auto _end = std::chrono::high_resolution_clock::now(); \
            auto _d = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n'; \
        } \
    }

}























































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class UnaryFunction
>
inline
cudaError_t transform(InputIterator input,
                     OutputIterator output,
                     const size_t size,
                     UnaryFunction transform_op,
                     const cudaStream_t stream = 0,
                     bool debug_synchronous = false)
{
    if( size == size_t(0) )
        return cudaSuccess;

    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using result_type = typename ::rocprim::detail::invoke_result<UnaryFunction, input_type>::type;


    using config = detail::default_or_custom_config<
        Config,
        detail::default_transform_config<ROCPRIM_TARGET_ARCH, result_type>
    >;

    static constexpr unsigned int block_size = config::block_size;
    static constexpr unsigned int items_per_thread = config::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;


    std::chrono::high_resolution_clock::time_point start;

    static constexpr auto size_limit = config::size_limit;
    static constexpr auto number_of_blocks_limit
        = ::rocprim::max<size_t>(size_limit / items_per_block, 1);

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "number of blocks limit " << number_of_blocks_limit << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    static constexpr auto aligned_size_limit = number_of_blocks_limit * items_per_block;


    const auto number_of_launch = (size + aligned_size_limit - 1) / aligned_size_limit;
    for(size_t i = 0, offset = 0; i < number_of_launch; ++i, offset += aligned_size_limit) {
        const auto current_size = std::min(size - offset, aligned_size_limit);
        const auto current_blocks = (current_size + items_per_block - 1) / items_per_block;

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        detail::transform_kernel<
                block_size, items_per_thread, result_type,
                InputIterator, OutputIterator, UnaryFunction
            >
            <<<dim3(current_blocks), dim3(block_size), 0, stream>>>(
            input + offset, current_size, output + offset, transform_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("transform_kernel", current_size, start);
    }

    return cudaSuccess;
}



























































template<
    class Config = default_config,
    class InputIterator1,
    class InputIterator2,
    class OutputIterator,
    class BinaryFunction
>
inline
cudaError_t transform(InputIterator1 input1,
                     InputIterator2 input2,
                     OutputIterator output,
                     const size_t size,
                     BinaryFunction transform_op,
                     const cudaStream_t stream = 0,
                     bool debug_synchronous = false)
{
    using value_type1 = typename std::iterator_traits<InputIterator1>::value_type;
    using value_type2 = typename std::iterator_traits<InputIterator2>::value_type;
    return transform<Config>(
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(input1, input2)), output,
        size, detail::unpack_binary_op<value_type1, value_type2, BinaryFunction>(transform_op),
        stream, debug_synchronous
    );
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR




END_ROCPRIM_NAMESPACE

#endif
