



















#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_BY_KEY_HPP_

#include "detail/device_scan_by_key.hpp"
#include "detail/lookback_scan_state.hpp"
#include "detail/ordered_block_id.hpp"

#include "config_types.hpp"
#include "device_scan_by_key_config.hpp"

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"
#include "../types/future_value.hpp"
#include "../types/tuple.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <iterator>
#include <type_traits>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

    template <bool Exclusive,
              typename Config,
              typename KeyInputIterator,
              typename InputIterator,
              typename OutputIterator,
              typename InitialValueType,
              typename CompareFunction,
              typename BinaryFunction,
              typename LookbackScanState,
              typename ResultType>
    void __global__ __launch_bounds__(Config::block_size) device_scan_by_key_kernel(
        const KeyInputIterator                          keys,
        const InputIterator                             values,
        const OutputIterator                            output,
        const InitialValueType                          initial_value,
        const CompareFunction                           compare,
        const BinaryFunction                            scan_op,
        const LookbackScanState                         scan_state,
        const size_t                                    size,
        const size_t                                    starting_block,
        const size_t                                    number_of_blocks,
        const ordered_block_id<unsigned int>            ordered_bid,
        const ::rocprim::tuple<ResultType, bool>* const previous_last_value)
    {
        device_scan_by_key_kernel_impl<Exclusive, Config>(keys,
                                                          values,
                                                          output,
                                                          get_input_value(initial_value),
                                                          compare,
                                                          scan_op,
                                                          scan_state,
                                                          size,
                                                          starting_block,
                                                          number_of_blocks,
                                                          ordered_bid,
                                                          previous_last_value);
    }

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    do                                                                                           \
    {                                                                                            \
        auto _error = cudaGetLastError();                                                         \
        if(_error != cudaSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = cudaStreamSynchronize(stream);                                         \
            if(__error != cudaSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    } while(false)

    template <bool Exclusive,
              typename Config,
              typename KeysInputIterator,
              typename InputIterator,
              typename OutputIterator,
              typename InitValueType,
              typename BinaryFunction,
              typename CompareFunction>
    inline cudaError_t scan_by_key_impl(void* const           temporary_storage,
                                       size_t&               storage_size,
                                       KeysInputIterator     keys,
                                       InputIterator         input,
                                       OutputIterator        output,
                                       const InitValueType   initial_value,
                                       const size_t          size,
                                       const BinaryFunction  scan_op,
                                       const CompareFunction compare,
                                       const cudaStream_t     stream,
                                       const bool            debug_synchronous)
    {
        using config               = Config;
        using real_init_value_type = input_type_t<InitValueType>;

        using wrapped_type = ::rocprim::tuple<real_init_value_type, bool>;

        using scan_state_type            = detail::lookback_scan_state<wrapped_type>;
        using scan_state_with_sleep_type = detail::lookback_scan_state<wrapped_type, true>;
        using ordered_block_id_type      = detail::ordered_block_id<unsigned int>;

        constexpr unsigned int block_size       = config::block_size;
        constexpr unsigned int items_per_thread = config::items_per_thread;
        constexpr auto         items_per_block  = block_size * items_per_thread;

        static constexpr size_t size_limit = config::size_limit;
        static constexpr size_t aligned_size_limit
            = ::rocprim::max<size_t>(size_limit - size_limit % items_per_block, items_per_block);

        const size_t limited_size     = std::min<size_t>(size, aligned_size_limit);
        const bool   use_limited_size = limited_size == aligned_size_limit;


        const unsigned int number_of_blocks = ceiling_div(limited_size, items_per_block);


        const size_t scan_state_bytes
            = align_size(scan_state_type::get_storage_size(number_of_blocks));
        if(temporary_storage == nullptr)
        {
            const size_t ordered_block_id_bytes
                = align_size(ordered_block_id_type::get_storage_size(), alignof(wrapped_type));


            storage_size = scan_state_bytes + ordered_block_id_bytes
                           + (use_limited_size ? sizeof(wrapped_type) : 0);

            return cudaSuccess;
        }

        if(number_of_blocks == 0u)
        {
            return cudaSuccess;
        }

        bool use_sleep;
        if(const cudaError_t error = is_sleep_scan_state_used(use_sleep))
        {
            return error;
        }



        auto with_scan_state
            = [use_sleep,
               scan_state            = scan_state_type::create(temporary_storage, number_of_blocks),
               scan_state_with_sleep = scan_state_with_sleep_type::create(
                   temporary_storage, number_of_blocks)](auto&& func) mutable -> decltype(auto) {
            if(use_sleep)
            {
                return func(scan_state_with_sleep);
            }
            else
            {
                return func(scan_state);
            }
        };


        auto* const ptr         = static_cast<char*>(temporary_storage);
        const auto  ordered_bid = ordered_block_id_type::create(
             reinterpret_cast<ordered_block_id_type::id_type*>(ptr + scan_state_bytes));


        auto* const previous_last_value
            = use_limited_size
                  ? reinterpret_cast<wrapped_type*>(ptr + storage_size - sizeof(wrapped_type))
                  : nullptr;


        const auto   total_number_of_blocks = ceiling_div(size, items_per_block);
        const size_t number_of_launch       = ceiling_div(size, limited_size);

        if(debug_synchronous)
        {
            std::cout << "----------------------------------\n";
            std::cout << "size:               " << size << '\n';
            std::cout << "aligned_size_limit: " << aligned_size_limit << '\n';
            std::cout << "use_limited_size:   " << std::boolalpha << use_limited_size << '\n';
            std::cout << "number_of_launch:   " << number_of_launch << '\n';
            std::cout << "block_size:         " << block_size << '\n';
            std::cout << "items_per_block:    " << items_per_block << '\n';
            std::cout << "----------------------------------\n";
        }

        for(size_t i = 0, offset = 0; i < number_of_launch; i++, offset += limited_size)
        {
            const size_t current_size = std::min<size_t>(size - offset, limited_size);
            const auto   scan_blocks  = ceiling_div(current_size, items_per_block);
            const auto init_grid_size = ceiling_div(scan_blocks, block_size);


            std::chrono::high_resolution_clock::time_point start;
            if(debug_synchronous)
            {
                std::cout << "index:            " << i << '\n';
                std::cout << "current_size:     " << current_size << '\n';
                std::cout << "number of blocks: " << scan_blocks << '\n';

                start = std::chrono::high_resolution_clock::now();
            }

            with_scan_state([&](const auto scan_state) {
                init_lookback_scan_state_kernel<<<
                                   dim3(init_grid_size),
                                   dim3(block_size),
                                   0,
                                   stream>>>(
                                   scan_state,
                                   scan_blocks,
                                   ordered_bid,
                                   number_of_blocks - 1,
                                   i > 0 ? previous_last_value : nullptr);
            });
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
                "init_lookback_scan_state_kernel", scan_blocks, start);

            if(debug_synchronous)
            {
                start = std::chrono::high_resolution_clock::now();
            }
            with_scan_state([&](auto& scan_state) {
                device_scan_by_key_kernel<Exclusive, config><<<
                                   dim3(scan_blocks),
                                   dim3(block_size),
                                   0,
                                   stream>>>(
                                   keys + offset,
                                   input + offset,
                                   output + offset,
                                   initial_value,
                                   compare,
                                   scan_op,
                                   scan_state,
                                   size,
                                   i * number_of_blocks,
                                   total_number_of_blocks,
                                   ordered_bid,
                                   i > 0 ? previous_last_value : nullptr);
            });
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(
                "device_scan_by_key_kernel", current_size, start);
        }
        return cudaSuccess;
    }

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
}






























































































template <typename Config = default_config,
          typename KeysInputIterator,
          typename ValuesInputIterator,
          typename ValuesOutputIterator,
          typename BinaryFunction
          = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
          typename KeyCompareFunction
          = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>>
inline cudaError_t inclusive_scan_by_key(void* const                temporary_storage,
                                        size_t&                    storage_size,
                                        const KeysInputIterator    keys_input,
                                        const ValuesInputIterator  values_input,
                                        const ValuesOutputIterator values_output,
                                        const size_t               size,
                                        const BinaryFunction       scan_op = BinaryFunction(),
                                        const KeyCompareFunction   key_compare_op
                                        = KeyCompareFunction(),
                                        const cudaStream_t stream            = 0,
                                        const bool        debug_synchronous = false)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;


    using config = detail::default_or_custom_config<
        Config,
        detail::default_scan_by_key_config<ROCPRIM_TARGET_ARCH, key_type, value_type>>;

    return detail::scan_by_key_impl<false, config>(temporary_storage,
                                                   storage_size,
                                                   keys_input,
                                                   values_input,
                                                   values_output,
                                                   value_type(),
                                                   size,
                                                   scan_op,
                                                   key_compare_op,
                                                   stream,
                                                   debug_synchronous);
}































































































template <typename Config = default_config,
          typename KeysInputIterator,
          typename ValuesInputIterator,
          typename ValuesOutputIterator,
          typename InitialValueType,
          typename BinaryFunction
          = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>,
          typename KeyCompareFunction
          = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>>
inline cudaError_t exclusive_scan_by_key(void* const                temporary_storage,
                                        size_t&                    storage_size,
                                        const KeysInputIterator    keys_input,
                                        const ValuesInputIterator  values_input,
                                        const ValuesOutputIterator values_output,
                                        const InitialValueType     initial_value,
                                        const size_t               size,
                                        const BinaryFunction       scan_op = BinaryFunction(),
                                        const KeyCompareFunction   key_compare_op
                                        = KeyCompareFunction(),
                                        const cudaStream_t stream            = 0,
                                        const bool        debug_synchronous = false)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using real_init_value_type = detail::input_type_t<InitialValueType>;


    using config = detail::default_or_custom_config<
        Config,
        detail::default_scan_by_key_config<ROCPRIM_TARGET_ARCH, key_type, real_init_value_type>
    >;

    return detail::scan_by_key_impl<true, config>(temporary_storage,
                                                  storage_size,
                                                  keys_input,
                                                  values_input,
                                                  values_output,
                                                  initial_value,
                                                  size,
                                                  scan_op,
                                                  key_compare_op,
                                                  stream,
                                                  debug_synchronous);
}




END_ROCPRIM_NAMESPACE

#endif
