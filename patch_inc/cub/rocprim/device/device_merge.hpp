



















#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "device_merge_config.hpp"
#include "detail/device_merge.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void partition_kernel(IndexIterator index,
                      KeysInputIterator1 keys_input1,
                      KeysInputIterator2 keys_input2,
                      const size_t input1_size,
                      const size_t input2_size,
                      const unsigned int spacing,
                      BinaryFunction compare_function)
{
    partition_kernel_impl(
        index, keys_input1, keys_input2, input1_size, input2_size,
        spacing, compare_function
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class IndexIterator,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void merge_kernel(IndexIterator index,
                  KeysInputIterator1 keys_input1,
                  KeysInputIterator2 keys_input2,
                  KeysOutputIterator keys_output,
                  ValuesInputIterator1 values_input1,
                  ValuesInputIterator2 values_input2,
                  ValuesOutputIterator values_output,
                  const size_t input1_size,
                  const size_t input2_size,
                  BinaryFunction compare_function)
{
    merge_kernel_impl<BlockSize, ItemsPerThread>(
        index, keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function
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
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction
>
inline
cudaError_t merge_impl(void * temporary_storage,
                      size_t& storage_size,
                      KeysInputIterator1 keys_input1,
                      KeysInputIterator2 keys_input2,
                      KeysOutputIterator keys_output,
                      ValuesInputIterator1 values_input1,
                      ValuesInputIterator2 values_input2,
                      ValuesOutputIterator values_output,
                      const size_t input1_size,
                      const size_t input2_size,
                      BinaryFunction compare_function,
                      const cudaStream_t stream,
                      bool debug_synchronous)

{
    using key_type = typename std::iterator_traits<KeysInputIterator1>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator1>::value_type;


    using config = detail::default_or_custom_config<
        Config,
        detail::default_merge_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    static constexpr unsigned int block_size = config::block_size;
    static constexpr unsigned int half_block = block_size / 2;
    static constexpr unsigned int items_per_thread = config::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;

    const unsigned int partitions = ((input1_size + input2_size) + items_per_block - 1) / items_per_block;
    const size_t partition_bytes = (partitions + 1) * sizeof(unsigned int);

    if(temporary_storage == nullptr)
    {

        storage_size = partition_bytes;
        return cudaSuccess;
    }

    if( partitions == 0u )
        return cudaSuccess;


    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = partitions;
    if(debug_synchronous)
    {
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    unsigned int * index = reinterpret_cast<unsigned int *>(temporary_storage);

    const unsigned partition_blocks = ((partitions + 1) + half_block - 1) / half_block;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    detail::partition_kernel
        <<<dim3(partition_blocks), dim3(half_block), 0, stream>>>(
        index, keys_input1, keys_input2, input1_size, input2_size,
        items_per_block, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", input1_size, start);

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    detail::merge_kernel<block_size, items_per_thread>
        <<<dim3(number_of_blocks), dim3(block_size), 0, stream>>>(
        index, keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("merge_kernel", input1_size, start);

    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}












































































template<
    class Config = default_config,
    class InputIterator1,
    class InputIterator2,
    class OutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<InputIterator1>::value_type>
>
inline
cudaError_t merge(void * temporary_storage,
                 size_t& storage_size,
                 InputIterator1 input1,
                 InputIterator2 input2,
                 OutputIterator output,
                 const size_t input1_size,
                 const size_t input2_size,
                 BinaryFunction compare_function = BinaryFunction(),
                 const cudaStream_t stream = 0,
                 bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    return detail::merge_impl<Config>(
        temporary_storage, storage_size,
        input1, input2, output,
        values, values, values,
        input1_size, input2_size, compare_function,
        stream, debug_synchronous
    );
}






























































































template<
    class Config = default_config,
    class KeysInputIterator1,
    class KeysInputIterator2,
    class KeysOutputIterator,
    class ValuesInputIterator1,
    class ValuesInputIterator2,
    class ValuesOutputIterator,
    class BinaryFunction = ::rocprim::less<typename std::iterator_traits<KeysInputIterator1>::value_type>
>
inline
cudaError_t merge(void * temporary_storage,
                 size_t& storage_size,
                 KeysInputIterator1 keys_input1,
                 KeysInputIterator2 keys_input2,
                 KeysOutputIterator keys_output,
                 ValuesInputIterator1 values_input1,
                 ValuesInputIterator2 values_input2,
                 ValuesOutputIterator values_output,
                 const size_t input1_size,
                 const size_t input2_size,
                 BinaryFunction compare_function = BinaryFunction(),
                 const cudaStream_t stream = 0,
                 bool debug_synchronous = false)
{
    return detail::merge_impl<Config>(
        temporary_storage, storage_size,
        keys_input1, keys_input2, keys_output,
        values_input1, values_input2, values_output,
        input1_size, input2_size, compare_function,
        stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
