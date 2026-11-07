



















#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_HPP_

#include <iostream>
#include <iterator>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "device_radix_sort_config.hpp"
#include "device_transform.hpp"
#include "detail/device_radix_sort.hpp"
#include "specialization/device_radix_single_sort.hpp"
#include "specialization/device_radix_merge_sort.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class Offset
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void fill_digit_counts_kernel(KeysInputIterator keys_input,
                              Offset size,
                              Offset * batch_digit_counts,
                              unsigned int bit,
                              unsigned int current_radix_bits,
                              unsigned int blocks_per_full_batch,
                              unsigned int full_batches)
{
    fill_digit_counts<BlockSize, ItemsPerThread, RadixBits, Descending>(
        keys_input, size,
        batch_digit_counts,
        bit, current_radix_bits,
        blocks_per_full_batch, full_batches
    );
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    class Offset
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void scan_batches_kernel(Offset * batch_digit_counts,
                         Offset * digit_counts,
                         unsigned int batches)
{
    scan_batches<BlockSize, ItemsPerThread, RadixBits>(batch_digit_counts, digit_counts, batches);
}

template<
    unsigned int RadixBits,
    class Offset
>
ROCPRIM_KERNEL
__launch_bounds__(ROCPRIM_DEFAULT_MAX_BLOCK_SIZE)
void scan_digits_kernel(Offset * digit_counts)
{
    scan_digits<RadixBits>(digit_counts);
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Offset
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void sort_and_scatter_kernel(KeysInputIterator keys_input,
                             KeysOutputIterator keys_output,
                             ValuesInputIterator values_input,
                             ValuesOutputIterator values_output,
                             Offset size,
                             const Offset * batch_digit_starts,
                             const Offset * digit_starts,
                             unsigned int bit,
                             unsigned int current_radix_bits,
                             unsigned int blocks_per_full_batch,
                             unsigned int full_batches)
{
    sort_and_scatter<BlockSize, ItemsPerThread, RadixBits, Descending>(
        keys_input, keys_output, values_input, values_output, size,
        batch_digit_starts, digit_starts,
        bit, current_radix_bits,
        blocks_per_full_batch, full_batches
    );
}

#ifndef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

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

#endif

template<
    class Config,
    unsigned int RadixBits,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Offset
>
inline
cudaError_t radix_sort_iteration(KeysInputIterator keys_input,
                                typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                KeysOutputIterator keys_output,
                                ValuesInputIterator values_input,
                                typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                ValuesOutputIterator values_output,
                                Offset size,
                                Offset * batch_digit_counts,
                                Offset * digit_counts,
                                bool from_input,
                                bool to_output,
                                unsigned int bit,
                                unsigned int end_bit,
                                unsigned int blocks_per_full_batch,
                                unsigned int full_batches,
                                unsigned int batches,
                                cudaStream_t stream,
                                bool debug_synchronous)
{
    constexpr unsigned int radix_size = 1 << RadixBits;



    const unsigned int current_radix_bits = ::rocprim::min(RadixBits, end_bit - bit);

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous)
    {
        std::cout << "RadixBits " << RadixBits << '\n';
        std::cout << "bit " << bit << '\n';
        std::cout << "current_radix_bits " << current_radix_bits << '\n';
    }

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    if(from_input)
    {
        fill_digit_counts_kernel<
                Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
            >
            <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
            keys_input, size,
            batch_digit_counts,
            bit, current_radix_bits,
            blocks_per_full_batch, full_batches
        );
    }
    else
    {
        if(to_output)
        {
            fill_digit_counts_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >
                <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
                keys_tmp, size,
                batch_digit_counts,
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
        else
        {
            fill_digit_counts_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >
                <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
                keys_output, size,
                batch_digit_counts,
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
    }
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("fill_digit_counts", size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    scan_batches_kernel<Config::scan::block_size, Config::scan::items_per_thread, RadixBits>
        <<<dim3(radix_size), dim3(Config::scan::block_size), 0, stream>>>(
        batch_digit_counts, digit_counts, batches
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_batches", radix_size * Config::scan::block_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    scan_digits_kernel<RadixBits>
        <<<dim3(1), dim3(radix_size), 0, stream>>>(
        digit_counts
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("scan_digits", radix_size, start)

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    if(from_input)
    {
        if(to_output)
        {
            sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >
                <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
                keys_input, keys_output, values_input, values_output, size,
                const_cast<const Offset *>(batch_digit_counts),
                const_cast<const Offset *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
        else
        {
            sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >
                <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
                keys_input, keys_tmp, values_input, values_tmp, size,
                const_cast<const Offset *>(batch_digit_counts),
                const_cast<const Offset *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
    }
    else
    {
        if(to_output)
        {
            sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >
                <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
                keys_tmp, keys_output, values_tmp, values_output, size,
                const_cast<const Offset *>(batch_digit_counts),
                const_cast<const Offset *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
        else
        {
            sort_and_scatter_kernel<
                    Config::sort::block_size, Config::sort::items_per_thread, RadixBits, Descending
                >
                <<<dim3(batches), dim3(Config::sort::block_size), 0, stream>>>(
                keys_output, keys_tmp, values_output, values_tmp, size,
                const_cast<const Offset *>(batch_digit_counts),
                const_cast<const Offset *>(digit_counts),
                bit, current_radix_bits,
                blocks_per_full_batch, full_batches
            );
        }
    }
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("sort_and_scatter", size, start)

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
cudaError_t radix_sort_single_impl(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 KeysOutputIterator keys_output,
                                 ValuesInputIterator values_input,
                                 ValuesOutputIterator values_output,
                                 unsigned int size,
                                 bool& is_result_in_output,
                                 unsigned int begin_bit,
                                 unsigned int end_bit,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using config = default_or_custom_config<
        Config,
        default_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    const size_t minimum_bytes = ::rocprim::detail::align_size(1);
    if(temporary_storage == nullptr)
    {
        storage_size = minimum_bytes;
        return cudaSuccess;
    }

    if( size == 0u )
        return cudaSuccess;

    if(debug_synchronous)
    {
        std::cout << "temporary_storage " << temporary_storage << '\n';
        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess) return error;
    }

    cudaError_t error = radix_sort_single<config, Descending>(
        keys_input, keys_output, values_input, values_output, size,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(error != cudaSuccess) return error;

    is_result_in_output = true;
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
cudaError_t radix_sort_merge_impl(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                 KeysOutputIterator keys_output,
                                 ValuesInputIterator values_input,
                                 typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                 ValuesOutputIterator values_output,
                                 unsigned int size,
                                 bool& is_result_in_output,
                                 unsigned int begin_bit,
                                 unsigned int end_bit,
                                 cudaStream_t stream,
                                 bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    using config = default_or_custom_config<
        Config,
        default_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    const bool with_double_buffer = keys_tmp != nullptr;
    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;

    const size_t minimum_bytes = ::rocprim::detail::align_size(1);
    if(temporary_storage == nullptr)
    {
        if(!with_double_buffer)
            storage_size = keys_bytes + values_bytes;
        else
            storage_size = minimum_bytes;
        return cudaSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "temporary_storage " << temporary_storage << '\n';
        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess) return error;
    }

    if(!with_double_buffer)
    {
        char * ptr = reinterpret_cast<char *>(temporary_storage);
        keys_tmp = reinterpret_cast<key_type *>(ptr);
        ptr += keys_bytes;
        values_tmp = with_values ? reinterpret_cast<value_type *>(ptr) : nullptr;
    }

    cudaError_t error = radix_sort_merge<config, Descending>(
        keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output, size,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(error != cudaSuccess) return error;

    is_result_in_output = true;
    return cudaSuccess;
}

template<class Size>
using offset_type_t = std::conditional_t<
    sizeof(Size) <= 4,
    unsigned int,
    size_t
>;

template<
    class Config,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Size
>
inline
cudaError_t radix_sort_iterations_impl(void * temporary_storage,
                                      size_t& storage_size,
                                      KeysInputIterator keys_input,
                                      typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                      KeysOutputIterator keys_output,
                                      ValuesInputIterator values_input,
                                      typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                      ValuesOutputIterator values_output,
                                      Size size,
                                      bool& is_result_in_output,
                                      unsigned int begin_bit,
                                      unsigned int end_bit,
                                      cudaStream_t stream,
                                      bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using offset_type = offset_type_t<Size>;

    using config = default_or_custom_config<
        Config,
        default_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;

    constexpr unsigned int max_radix_size = 1 << config::long_radix_bits;

    constexpr unsigned int scan_size = config::scan::block_size * config::scan::items_per_thread;
    constexpr unsigned int sort_size = config::sort::block_size * config::sort::items_per_thread;

    const unsigned int blocks = static_cast<unsigned int>(::rocprim::detail::ceiling_div(size, sort_size));
    const unsigned int blocks_per_full_batch = ::rocprim::detail::ceiling_div(blocks, scan_size);
    const unsigned int full_batches = blocks % scan_size != 0
        ? blocks % scan_size
        : scan_size;
    const unsigned int batches = (blocks_per_full_batch == 1 ? full_batches : scan_size);
    const bool with_double_buffer = keys_tmp != nullptr;

    const unsigned int bits = end_bit - begin_bit;
    const unsigned int iterations = ::rocprim::detail::ceiling_div(bits, config::long_radix_bits);
    const unsigned int radix_bits_diff = config::long_radix_bits - config::short_radix_bits;
    const unsigned int short_iterations = radix_bits_diff != 0
        ? ::rocprim::min(iterations, (config::long_radix_bits * iterations - bits) / std::max(1u, radix_bits_diff))
        : 0;
    const unsigned int long_iterations = iterations - short_iterations;

    const size_t batch_digit_counts_bytes =
        ::rocprim::detail::align_size(batches * max_radix_size * sizeof(offset_type));
    const size_t digit_counts_bytes = ::rocprim::detail::align_size(max_radix_size * sizeof(offset_type));
    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;
    if(temporary_storage == nullptr)
    {
        storage_size = batch_digit_counts_bytes + digit_counts_bytes;
        if(!with_double_buffer)
        {
            storage_size += keys_bytes + values_bytes;
        }
        return cudaSuccess;
    }

    if( size == 0u )
        return cudaSuccess;

    if(debug_synchronous)
    {
        std::cout << "scan_size " << scan_size << '\n';
        std::cout << "sort_size " << sort_size << '\n';
        std::cout << "blocks " << blocks << '\n';
        std::cout << "blocks_per_full_batch " << blocks_per_full_batch << '\n';
        std::cout << "full_batches " << full_batches << '\n';
        std::cout << "batches " << batches << '\n';
        std::cout << "iterations " << iterations << '\n';
        std::cout << "long_iterations " << long_iterations << '\n';
        std::cout << "short_iterations " << short_iterations << '\n';
        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess) return error;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    offset_type * batch_digit_counts = reinterpret_cast<offset_type *>(ptr);
    ptr += batch_digit_counts_bytes;
    offset_type * digit_counts = reinterpret_cast<offset_type *>(ptr);
    ptr += digit_counts_bytes;
    if(!with_double_buffer)
    {
        keys_tmp = reinterpret_cast<key_type *>(ptr);
        ptr += keys_bytes;
        values_tmp = with_values ? reinterpret_cast<value_type *>(ptr) : nullptr;
    }

    bool to_output = with_double_buffer || (iterations - 1) % 2 == 0;
    bool from_input = true;
    if(!with_double_buffer && to_output)
    {

        const bool keys_equal = ::rocprim::detail::are_iterators_equal(keys_input, keys_output);
        const bool values_equal = with_values && ::rocprim::detail::are_iterators_equal(values_input, values_output);
        if(keys_equal || values_equal)
        {
            cudaError_t error = ::rocprim::transform(
                keys_input, keys_tmp, size,
                ::rocprim::identity<key_type>(), stream, debug_synchronous
            );
            if(error != cudaSuccess) return error;

            if(with_values)
            {
                cudaError_t error = ::rocprim::transform(
                    values_input, values_tmp, size,
                    ::rocprim::identity<value_type>(), stream, debug_synchronous
                );
                if(error != cudaSuccess) return error;
            }

            from_input = false;
        }
    }

    unsigned int bit = begin_bit;
    for(unsigned int i = 0; i < long_iterations; i++)
    {
        cudaError_t error = radix_sort_iteration<config, config::long_radix_bits, Descending>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
            static_cast<offset_type>(size), batch_digit_counts, digit_counts,
            from_input, to_output,
            bit, end_bit,
            blocks_per_full_batch, full_batches, batches,
            stream, debug_synchronous
        );
        if(error != cudaSuccess) return error;

        is_result_in_output = to_output;
        from_input = false;
        to_output = !to_output;
        bit += config::long_radix_bits;
    }
    for(unsigned int i = 0; i < short_iterations; i++)
    {
        cudaError_t error = radix_sort_iteration<config, config::short_radix_bits, Descending>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
            static_cast<offset_type>(size), batch_digit_counts, digit_counts,
            from_input, to_output,
            bit, end_bit,
            blocks_per_full_batch, full_batches, batches,
            stream, debug_synchronous
        );
        if(error != cudaSuccess) return error;

        is_result_in_output = to_output;
        from_input = false;
        to_output = !to_output;
        bit += config::short_radix_bits;
    }

    return cudaSuccess;
}

template<
    class Config,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Size
>
inline
cudaError_t radix_sort_impl(void * temporary_storage,
                           size_t& storage_size,
                           KeysInputIterator keys_input,
                           typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                           ValuesOutputIterator values_output,
                           Size size,
                           bool& is_result_in_output,
                           unsigned int begin_bit,
                           unsigned int end_bit,
                           cudaStream_t stream,
                           bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;

    static_assert(
        std::is_same<key_type, typename std::iterator_traits<KeysOutputIterator>::value_type>::value,
        "KeysInputIterator and KeysOutputIterator must have the same value_type"
    );
    static_assert(
        std::is_same<value_type, typename std::iterator_traits<ValuesOutputIterator>::value_type>::value,
        "ValuesInputIterator and ValuesOutputIterator must have the same value_type"
    );

    using config = default_or_custom_config<
        Config,
        default_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    constexpr unsigned int single_sort_limit = config::sort_single::block_size * config::sort_single::items_per_thread;
    constexpr unsigned int merge_sort_limit = config::sort_merge::block_size * config::sort_merge::items_per_thread * config::merge_size_limit_blocks;

    if( size <= single_sort_limit )
    {
        return radix_sort_single_impl<Config, Descending>(
            temporary_storage,
            storage_size,
            keys_input,
            keys_output,
            values_input,
            values_output,
            static_cast<unsigned int>(size),
            is_result_in_output,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous
        );
    }
    else if( size <= merge_sort_limit )
    {
        return radix_sort_merge_impl<Config, Descending>(
            temporary_storage,
            storage_size,
            keys_input,
            keys_tmp,
            keys_output,
            values_input,
            values_tmp,
            values_output,
            static_cast<unsigned int>(size),
            is_result_in_output,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous
        );
    }
    else
    {
        return radix_sort_iterations_impl<Config, Descending>(
            temporary_storage,
            storage_size,
            keys_input,
            keys_tmp,
            keys_output,
            values_input,
            values_tmp,
            values_output,
            size,
            is_result_in_output,
            begin_bit,
            end_bit,
            stream,
            debug_synchronous
        );
    }
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}













































































template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t radix_sort_keys(void * temporary_storage,
                           size_t& storage_size,
                           KeysInputIterator keys_input,
                           KeysOutputIterator keys_output,
                           Size size,
                           unsigned int begin_bit = 0,
                           unsigned int end_bit = 8 * sizeof(Key),
                           cudaStream_t stream = 0,
                           bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool ignored;
    return detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}













































































template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t radix_sort_keys_desc(void * temporary_storage,
                                size_t& storage_size,
                                KeysInputIterator keys_input,
                                KeysOutputIterator keys_output,
                                Size size,
                                unsigned int begin_bit = 0,
                                unsigned int end_bit = 8 * sizeof(Key),
                                cudaStream_t stream = 0,
                                bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool ignored;
    return detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}





























































































template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t radix_sort_pairs(void * temporary_storage,
                            size_t& storage_size,
                            KeysInputIterator keys_input,
                            KeysOutputIterator keys_output,
                            ValuesInputIterator values_input,
                            ValuesOutputIterator values_output,
                            Size size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            cudaStream_t stream = 0,
                            bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool ignored;
    return detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

























































































template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class Size,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t radix_sort_pairs_desc(void * temporary_storage,
                                 size_t& storage_size,
                                 KeysInputIterator keys_input,
                                 KeysOutputIterator keys_output,
                                 ValuesInputIterator values_input,
                                 ValuesOutputIterator values_output,
                                 Size size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool ignored;
    return detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

















































































template<
    class Config = default_config,
    class Key,
    class Size
>
inline
cudaError_t radix_sort_keys(void * temporary_storage,
                           size_t& storage_size,
                           double_buffer<Key>& keys,
                           Size size,
                           unsigned int begin_bit = 0,
                           unsigned int end_bit = 8 * sizeof(Key),
                           cudaStream_t stream = 0,
                           bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool is_result_in_output;
    cudaError_t error = detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}

















































































template<
    class Config = default_config,
    class Key,
    class Size
>
inline
cudaError_t radix_sort_keys_desc(void * temporary_storage,
                                size_t& storage_size,
                                double_buffer<Key>& keys,
                                Size size,
                                unsigned int begin_bit = 0,
                                unsigned int end_bit = 8 * sizeof(Key),
                                cudaStream_t stream = 0,
                                bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    empty_type * values = nullptr;
    bool is_result_in_output;
    cudaError_t error = detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
    }
    return error;
}






























































































template<
    class Config = default_config,
    class Key,
    class Value,
    class Size
>
inline
cudaError_t radix_sort_pairs(void * temporary_storage,
                            size_t& storage_size,
                            double_buffer<Key>& keys,
                            double_buffer<Value>& values,
                            Size size,
                            unsigned int begin_bit = 0,
                            unsigned int end_bit = 8 * sizeof(Key),
                            cudaStream_t stream = 0,
                            bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool is_result_in_output;
    cudaError_t error = detail::radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}
























































































template<
    class Config = default_config,
    class Key,
    class Value,
    class Size
>
inline
cudaError_t radix_sort_pairs_desc(void * temporary_storage,
                                 size_t& storage_size,
                                 double_buffer<Key>& keys,
                                 double_buffer<Value>& values,
                                 Size size,
                                 unsigned int begin_bit = 0,
                                 unsigned int end_bit = 8 * sizeof(Key),
                                 cudaStream_t stream = 0,
                                 bool debug_synchronous = false)
{
    static_assert(std::is_integral<Size>::value, "Size must be an integral type.");
    bool is_result_in_output;
    cudaError_t error = detail::radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
    if(temporary_storage != nullptr && is_result_in_output)
    {
        keys.swap();
        values.swap();
    }
    return error;
}

END_ROCPRIM_NAMESPACE




#endif
