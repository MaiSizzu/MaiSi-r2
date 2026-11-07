



















#ifndef ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_SINGLE_SORT_HPP_
#define ROCPRIM_DEVICE_SPECIALIZATION_DEVICE_RADIX_SINGLE_SORT_HPP_

#include "../detail/device_radix_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

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
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    ROCPRIM_KERNEL
   __launch_bounds__(BlockSize)
   void sort_single_kernel(KeysInputIterator    keys_input,
                           KeysOutputIterator   keys_output,
                           ValuesInputIterator  values_input,
                           ValuesOutputIterator values_output,
                           unsigned int         size,
                           unsigned int         bit,
                           unsigned int         current_radix_bits)
   {
       sort_single<BlockSize, ItemsPerThread, Descending>(
           keys_input, keys_output,
           values_input, values_output,
           size, bit, current_radix_bits
       );
   }

    template<
        unsigned int BlockSize,
        unsigned int ItemsPerThread,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single(KeysInputIterator keys_input,
                                KeysOutputIterator keys_output,
                                ValuesInputIterator values_input,
                                ValuesOutputIterator values_output,
                                unsigned int size,
                                unsigned int bit,
                                unsigned int end_bit,
                                cudaStream_t stream,
                                bool debug_synchronous)
    {
        const unsigned int current_radix_bits = end_bit - bit;

        std::chrono::high_resolution_clock::time_point start;
        if(debug_synchronous)
        {
            std::cout << "BlockSize " << BlockSize << '\n';
            std::cout << "ItemsPerThread " << ItemsPerThread << '\n';
            std::cout << "bit " << bit << '\n';
            std::cout << "current_radix_bits " << current_radix_bits << '\n';
        }

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();

        sort_single_kernel<
                BlockSize, ItemsPerThread, Descending
            >
            <<<dim3(1), dim3(BlockSize), 0, stream>>>(
            keys_input, keys_output, values_input, values_output,
            size, bit, current_radix_bits
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("radix_sort_single", size, start)

        return cudaSuccess;
    }


    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit64(KeysInputIterator keys_input,
                                         KeysOutputIterator keys_output,
                                         ValuesInputIterator values_input,
                                         ValuesOutputIterator values_output,
                                         unsigned int size,
                                         unsigned int bit,
                                         unsigned int end_bit,
                                         cudaStream_t stream,
                                         bool debug_synchronous)
    {
        return radix_sort_single<64U, 1U, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit128(KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          ValuesInputIterator values_input,
                                          ValuesOutputIterator values_output,
                                          unsigned int size,
                                          unsigned int bit,
                                          unsigned int end_bit,
                                          cudaStream_t stream,
                                          bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 64U )
            return radix_sort_single_limit64<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<64U, 2U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit192(KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          ValuesInputIterator values_input,
                                          ValuesOutputIterator values_output,
                                          unsigned int size,
                                          unsigned int bit,
                                          unsigned int end_bit,
                                          cudaStream_t stream,
                                          bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 128U )
            return radix_sort_single_limit128<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<64U, 3U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit256(KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          ValuesInputIterator values_input,
                                          ValuesOutputIterator values_output,
                                          unsigned int size,
                                          unsigned int bit,
                                          unsigned int end_bit,
                                          cudaStream_t stream,
                                          bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 192U )
            return radix_sort_single_limit192<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<64U, 4U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit320(KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          ValuesInputIterator values_input,
                                          ValuesOutputIterator values_output,
                                          unsigned int size,
                                          unsigned int bit,
                                          unsigned int end_bit,
                                          cudaStream_t stream,
                                          bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 256U )
            return radix_sort_single_limit256<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<64U, 5U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit512(KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          ValuesInputIterator values_input,
                                          ValuesOutputIterator values_output,
                                          unsigned int size,
                                          unsigned int bit,
                                          unsigned int end_bit,
                                          cudaStream_t stream,
                                          bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 320U )
            return radix_sort_single_limit320<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 2U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit768(KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          ValuesInputIterator values_input,
                                          ValuesOutputIterator values_output,
                                          unsigned int size,
                                          unsigned int bit,
                                          unsigned int end_bit,
                                          cudaStream_t stream,
                                          bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 512U )
            return radix_sort_single_limit512<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 3U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit1024(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 768U )
            return radix_sort_single_limit768<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 4U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }



    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit1536(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 1024U )
            return radix_sort_single_limit1024<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 6U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit2048(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 1536U )
            return radix_sort_single_limit1536<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 8U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit2560(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 2048U )
            return radix_sort_single_limit2048<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 10U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit3072(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 2560U )
            return radix_sort_single_limit2560<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 12U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit3584(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 3072U )
            return radix_sort_single_limit3072<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 14U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    cudaError_t radix_sort_single_limit4096(KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int bit,
                                           unsigned int end_bit,
                                           cudaStream_t stream,
                                           bool debug_synchronous)
    {
        if( !Config::force_single_kernel_config && size <= 3584U )
            return radix_sort_single_limit3584<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<256U, 16U, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 64U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit64<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 64U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 128U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit128<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 128U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 192U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit192<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 192U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 256U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit256<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 256U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 320U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit320<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 320U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 512U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit512<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 512U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 768U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit768<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 768U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 1024U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit1024<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 1024U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 1536U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit1536<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 1536U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 2048U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit2048<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 2048U) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 2560U,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit2560<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 2560) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 3072,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit3072<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 3072) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 3584,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit3584<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 3584) &&
            Config::sort_single::items_per_thread * Config::sort_single::block_size <= 4096,
            cudaError_t
        >::type
    {
        return radix_sort_single_limit4096<Config, Descending>(
            keys_input, keys_output, values_input, values_output,
            size, bit, end_bit, stream, debug_synchronous
        );
    }

    template<
        class Config,
        bool Descending,
        class KeysInputIterator,
        class KeysOutputIterator,
        class ValuesInputIterator,
        class ValuesOutputIterator
    >
    inline
    auto radix_sort_single(KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           ValuesOutputIterator values_output,
                           unsigned int size,
                           unsigned int bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
        -> typename std::enable_if<
            (Config::sort_single::items_per_thread * Config::sort_single::block_size > 4096),
            cudaError_t
        >::type
    {
        if( size < 4096 )
            return radix_sort_single_limit4096<Config, Descending>(
                keys_input, keys_output, values_input, values_output,
                size, bit, end_bit, stream, debug_synchronous
            );
        else
            return radix_sort_single<
                Config::sort_single::block_size,
                Config::sort_single::items_per_thread,
                Descending
            >(
                    keys_input, keys_output, values_input, values_output,
                    size, bit, end_bit, stream, debug_synchronous
            );
    }

}

END_ROCPRIM_NAMESPACE

#endif
