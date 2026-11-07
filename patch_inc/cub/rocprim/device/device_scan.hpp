



















#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../types/future_value.hpp"
#include "../detail/various.hpp"

#include "device_scan_config.hpp"
#include "device_transform.hpp"
#include "detail/device_scan_common.hpp"
#include "detail/device_scan_lookback.hpp"
#include "detail/device_scan_reduce_then_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{


template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class InitValueType
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void single_scan_kernel(InputIterator input,
                        const size_t size,
                        const InitValueType initial_value,
                        OutputIterator output,
                        BinaryFunction scan_op)
{
    single_scan_kernel_impl<Exclusive, Config>(
        input, size, get_input_value(initial_value), output, scan_op
    );
}





template<
    class Config,
    class InputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void block_reduce_kernel(InputIterator input,
                         BinaryFunction scan_op,
                         ResultType * block_prefixes)
{
    block_reduce_kernel_impl<Config>(
        input, scan_op, block_prefixes
    );
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class InitValueType
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void final_scan_kernel(InputIterator input,
                       const size_t size,
                       OutputIterator output,
                       const InitValueType initial_value,
                       BinaryFunction scan_op,
                       input_type_t<InitValueType>* block_prefixes,
                       input_type_t<InitValueType>* previous_last_element = nullptr,
                       input_type_t<InitValueType>* new_last_element = nullptr,
                       bool override_first_value = false,
                       bool save_last_value = false)
{
    final_scan_kernel_impl<Exclusive, Config>(
        input, size, output, get_input_value(initial_value),
        scan_op, block_prefixes,
        previous_last_element, new_last_element,
        override_first_value, save_last_value
    );
}



template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class InitValueType,
    class LookBackScanState
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void lookback_scan_kernel(InputIterator input,
                          OutputIterator output,
                          const size_t size,
                          const InitValueType initial_value,
                          BinaryFunction scan_op,
                          LookBackScanState lookback_scan_state,
                          const unsigned int number_of_blocks,
                          ordered_block_id<unsigned int> ordered_bid,
                          input_type_t<InitValueType>* previous_last_element = nullptr,
                          input_type_t<InitValueType>* new_last_element = nullptr,
                          bool override_first_value = false,
                          bool save_last_value = false)
{
    lookback_scan_kernel_impl<Exclusive, Config>(
        input, output, size, get_input_value(initial_value), scan_op,
        lookback_scan_state, number_of_blocks, ordered_bid,
        previous_last_element, new_last_element,
        override_first_value, save_last_value
    );
}

#define ROCPRIM_DETAIL_HIP_SYNC(name, size, start) \
    if(debug_synchronous) \
    { \
        std::cout << name << "(" << size << ")"; \
        auto error = cudaStreamSynchronize(stream); \
        if(error != cudaSuccess) return error; \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
        std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
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
    class InitValueType,
    class BinaryFunction
>
inline
auto scan_impl(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               OutputIterator output,
               const InitValueType initial_value,
               const size_t size,
               BinaryFunction scan_op,
               const cudaStream_t stream,
               bool debug_synchronous)
    -> typename std::enable_if<!Config::use_lookback, cudaError_t>::type
{
    using config = Config;
    using real_init_value_type = input_type_t<InitValueType>;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    static constexpr size_t size_limit = config::size_limit;
    static constexpr size_t aligned_size_limit = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);
    size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;
    size_t nested_prefixes_size_bytes = scan_get_temporary_storage_bytes<real_init_value_type>(limited_size, items_per_block);


    if(temporary_storage == nullptr)
    {
        storage_size = nested_prefixes_size_bytes;

        if(use_limited_size)
            storage_size += 4 * sizeof(real_init_value_type);



        storage_size = storage_size == 0 ? 4 : storage_size;
        return cudaSuccess;
    }


    std::chrono::high_resolution_clock::time_point start;

    auto number_of_blocks = (size + items_per_block - 1)/items_per_block;

    if( number_of_blocks == 0u )
        return cudaSuccess;

    if(number_of_blocks > 1)
    {
        unsigned int number_of_launch = (size + limited_size - 1)/limited_size;
        for (size_t i = 0, offset = 0; i < number_of_launch; i++, offset+=limited_size )
        {
            size_t current_size = std::min<size_t>(size - offset, limited_size);
            number_of_blocks = (current_size + items_per_block - 1)/items_per_block;
            if(debug_synchronous)
            {
                std::cout << "use_limited_size " << use_limited_size << '\n';
                std::cout << "number_of_launch " << number_of_launch << '\n';
                std::cout << "inex " << i << '\n';
                std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
                std::cout << "size " << current_size << '\n';
                std::cout << "block_size " << block_size << '\n';
                std::cout << "number of blocks " << number_of_blocks << '\n';
                std::cout << "items_per_block " << items_per_block << '\n';
                std::cout.flush();
            }


            char * ptr = reinterpret_cast<char *>(temporary_storage);
            real_init_value_type* block_prefixes = reinterpret_cast<real_init_value_type*>(ptr);
            real_init_value_type* previous_last_element = nullptr;
            real_init_value_type* new_last_element = nullptr;
            if(use_limited_size)
            {
                ptr += nested_prefixes_size_bytes;
                previous_last_element = reinterpret_cast<real_init_value_type*>(ptr);

                ptr += sizeof(real_init_value_type);
                new_last_element = reinterpret_cast<real_init_value_type*>(ptr);
            }



            auto grid_size = number_of_blocks - 1;
            if( grid_size != 0 )
            {
                if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
                detail::block_reduce_kernel<
                        config, InputIterator, BinaryFunction, real_init_value_type
                    >
                    <<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                    input + offset, scan_op, block_prefixes
                );
                ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("block_reduce_kernel", current_size, start)

                if( !Exclusive && i > 0 )
                {
                    cudaError_t error = ::rocprim::transform(
                        previous_last_element, block_prefixes, block_prefixes, 1,
                        scan_op, stream, debug_synchronous
                    );
                    if(error != cudaSuccess) return error;
                }





                void * nested_temp_storage = static_cast<void*>(block_prefixes + number_of_blocks);
                auto nested_temp_storage_size = storage_size - (number_of_blocks * sizeof(real_init_value_type));

                if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
                auto error = scan_impl<false, config>(
                    nested_temp_storage,
                    nested_temp_storage_size,
                    block_prefixes,
                    block_prefixes,
                    real_init_value_type(),
                    number_of_blocks,
                    scan_op,
                    stream,
                    debug_synchronous
                );
                if(error != cudaSuccess) return error;
                ROCPRIM_DETAIL_HIP_SYNC("nested_device_scan", number_of_blocks, start);

            }


            grid_size = number_of_blocks;
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            detail::final_scan_kernel<
                    Exclusive,
                    config,
                    InputIterator, OutputIterator,
                    BinaryFunction, InitValueType
                >
                <<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                input + offset,
                current_size,
                output + offset,
                initial_value,
                scan_op,
                block_prefixes,
                previous_last_element,
                new_last_element,
                i != size_t(0) && ((!Exclusive && number_of_blocks == 1) || Exclusive),
                number_of_launch > 1
            );
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("final_scan_kernel", size, start);


            if(number_of_launch > 1)
            {
                cudaError_t error = ::rocprim::transform(
                    new_last_element, previous_last_element, 1,
                    ::rocprim::identity<real_init_value_type>(),
                    stream, debug_synchronous
                );
                if(error != cudaSuccess) return error;
            }
        }
    }
    else
    {
        if(debug_synchronous)
        {
            std::cout << "block_size " << block_size << '\n';
            std::cout << "number of blocks " << number_of_blocks << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
        }

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        detail::single_scan_kernel<
                Exclusive,
                config,
                InputIterator, OutputIterator, BinaryFunction
            >
            <<<dim3(1), dim3(block_size), 0, stream>>>(
            input, size, initial_value, output, scan_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("single_scan_kernel", size, start);
    }
    return cudaSuccess;
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction
>
inline
auto scan_impl(void * temporary_storage,
               size_t& storage_size,
               InputIterator input,
               OutputIterator output,
               const InitValueType initial_value,
               const size_t size,
               BinaryFunction scan_op,
               const cudaStream_t stream,
               bool debug_synchronous)
    -> typename std::enable_if<Config::use_lookback, cudaError_t>::type
{
    using config = Config;
    using real_init_value_type = input_type_t<InitValueType>;

    using scan_state_type = detail::lookback_scan_state<real_init_value_type>;
    using scan_state_with_sleep_type = detail::lookback_scan_state<real_init_value_type, true>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;

    constexpr unsigned int block_size = config::block_size;
    constexpr unsigned int items_per_thread = config::items_per_thread;
    constexpr auto items_per_block = block_size * items_per_thread;

    static constexpr size_t size_limit = config::size_limit;
    static constexpr size_t aligned_size_limit = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);
    size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;

    unsigned int number_of_blocks = (limited_size + items_per_block - 1)/items_per_block;


    size_t scan_state_bytes = ::rocprim::detail::align_size(

        scan_state_type::get_storage_size(number_of_blocks)
    );
    size_t ordered_block_id_bytes = ordered_block_id_type::get_storage_size();
    if(temporary_storage == nullptr)
    {

        storage_size = scan_state_bytes + ordered_block_id_bytes;

        if(use_limited_size)
            storage_size += 2 * sizeof(real_init_value_type);

        return cudaSuccess;
    }


    std::chrono::high_resolution_clock::time_point start;

    if( number_of_blocks == 0u )
        return cudaSuccess;

    if(number_of_blocks > 1 || use_limited_size)
    {

        auto scan_state = scan_state_type::create(temporary_storage, number_of_blocks);
        auto scan_state_with_sleep = scan_state_with_sleep_type::create(temporary_storage, number_of_blocks);

        auto ptr = reinterpret_cast<char*>(temporary_storage);
        auto ordered_bid = ordered_block_id_type::create(
            reinterpret_cast<ordered_block_id_type::id_type*>(ptr + scan_state_bytes)
        );


        real_init_value_type* previous_last_element = nullptr;
        real_init_value_type* new_last_element = nullptr;
        if(use_limited_size)
        {
            ptr += storage_size - sizeof(real_init_value_type);
            new_last_element = reinterpret_cast<real_init_value_type*>(ptr);
            ptr -= sizeof(real_init_value_type);
            previous_last_element = reinterpret_cast<real_init_value_type*>(ptr);
        }

        cudaDeviceProp prop;
        int deviceId;
        static_cast<void>(cudaGetDevice(&deviceId));
        static_cast<void>(cudaGetDeviceProperties(&prop, deviceId));

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();


        int asicRevision = 0;


        size_t number_of_launch = (size + limited_size - 1)/limited_size;
        for (size_t i = 0, offset = 0; i < number_of_launch; i++, offset+=limited_size )
        {
            size_t current_size = std::min<size_t>(size - offset, limited_size);
            number_of_blocks = (current_size + items_per_block - 1)/items_per_block;
            auto grid_size = (number_of_blocks + block_size - 1)/block_size;

            if(debug_synchronous)
            {
                std::cout << "use_limited_size " << use_limited_size << '\n';
                std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
                std::cout << "number_of_launch " << number_of_launch << '\n';
                std::cout << "index " << i << '\n';
                std::cout << "size " << current_size << '\n';
                std::cout << "block_size " << block_size << '\n';
                std::cout << "number of blocks " << number_of_blocks << '\n';
                std::cout << "items_per_block " << items_per_block << '\n';
            }


            
                init_lookback_scan_state_kernel<scan_state_type>
                    <<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                    scan_state, number_of_blocks, ordered_bid
                );
            
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel", number_of_blocks, start)

            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            grid_size = number_of_blocks;

            
                if(debug_synchronous)
                {
                    std::cout << "use_limited_size " << use_limited_size << '\n';
                    std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
                    std::cout << "size " << current_size << '\n';
                    std::cout << "block_size " << block_size << '\n';
                    std::cout << "number of blocks " << number_of_blocks << '\n';
                    std::cout << "items_per_block " << items_per_block << '\n';
                }

                lookback_scan_kernel<
                        Exclusive,
                        config,
                        InputIterator, OutputIterator,
                        BinaryFunction, InitValueType, scan_state_type
                    >
                    <<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                    input + offset, output + offset, current_size, initial_value,
                    scan_op, scan_state, number_of_blocks, ordered_bid,
                    previous_last_element, new_last_element,
                    i != size_t(0), number_of_launch > 1
                );
            
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("lookback_scan_kernel", current_size, start)


            if(number_of_launch > 1)
            {
                cudaError_t error = ::rocprim::transform(
                    new_last_element, previous_last_element, 1,
                    ::rocprim::identity<real_init_value_type>(),
                    stream, debug_synchronous
                );
                if(error != cudaSuccess) return error;
            }
        }
    }
    else
    {
        if(debug_synchronous)
        {
            std::cout << "size " << size << '\n';
            std::cout << "block_size " << block_size << '\n';
            std::cout << "number of blocks " << number_of_blocks << '\n';
            std::cout << "items_per_block " << items_per_block << '\n';
        }

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        single_scan_kernel<
                Exclusive,
                config,
                InputIterator, OutputIterator, BinaryFunction
            >
            <<<dim3(1), dim3(block_size), 0, stream>>>(
            input, size, initial_value, output, scan_op
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("single_scan_kernel", size, start);
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
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t inclusive_scan(void * temporary_storage,
                          size_t& storage_size,
                          InputIterator input,
                          OutputIterator output,
                          const size_t size,
                          BinaryFunction scan_op = BinaryFunction(),
                          const cudaStream_t stream = 0,
                          bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;


    using config = detail::default_or_custom_config<
        Config,
        detail::default_scan_config<ROCPRIM_TARGET_ARCH, input_type>
        >;

    return detail::scan_impl<false, config>(
        temporary_storage, storage_size,

        input, output, input_type(), size,
        scan_op, stream, debug_synchronous
    );
}






















































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class InitValueType,
    class BinaryFunction = ::rocprim::plus<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t exclusive_scan(void * temporary_storage,
                          size_t& storage_size,
                          InputIterator input,
                          OutputIterator output,
                          const InitValueType initial_value,
                          const size_t size,
                          BinaryFunction scan_op = BinaryFunction(),
                          const cudaStream_t stream = 0,
                          bool debug_synchronous = false)
{
    using real_init_value_type = detail::input_type_t<InitValueType>;


    using config = detail::default_or_custom_config<
        Config,
        detail::default_scan_config<ROCPRIM_TARGET_ARCH, real_init_value_type>
    >;

    return detail::scan_impl<true, config>(
        temporary_storage, storage_size,
        input, output, initial_value, size,
        scan_op, stream, debug_synchronous
    );
}




END_ROCPRIM_NAMESPACE

#endif
