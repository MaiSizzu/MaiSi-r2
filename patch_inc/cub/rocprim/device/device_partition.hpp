



















#ifndef ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_
#define ROCPRIM_DEVICE_DEVICE_PARTITION_HPP_

#include <algorithm>
#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../functional.hpp"
#include "../types.hpp"
#include "../type_traits.hpp"
#include "../detail/various.hpp"

#include "device_select_config.hpp"
#include "detail/device_scan_common.hpp"
#include "detail/device_partition.hpp"
#include "device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

template<
    select_method SelectMethod,
    bool OnlySelected,
    class Config,
    class KeyIterator,
    class ValueIterator,
    class FlagIterator,
    class OutputKeyIterator,
    class OutputValueIterator,
    class InequalityOp,
    class OffsetLookbackScanState,
    class... UnaryPredicates
>
ROCPRIM_KERNEL
__launch_bounds__(Config::block_size)
void partition_kernel(KeyIterator keys_input,
                      ValueIterator values_input,
                      FlagIterator flags,
                      OutputKeyIterator keys_output,
                      OutputValueIterator values_output,
                      size_t* selected_count,
                      size_t* prev_selected_count,
                      const size_t size,
                      InequalityOp inequality_op,
                      OffsetLookbackScanState offset_scan_state,
                      const unsigned int number_of_blocks,
                      ordered_block_id<unsigned int> ordered_bid,
                      UnaryPredicates... predicates)
{
    partition_kernel_impl<SelectMethod, OnlySelected, Config>(
        keys_input, values_input, flags, keys_output, values_output, selected_count, prev_selected_count, 
        size, inequality_op, offset_scan_state, number_of_blocks, ordered_bid, predicates...
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

    select_method SelectMethod,

    bool OnlySelected,
    class Config,
    class OffsetT,
    class KeyIterator,
    class ValueIterator,
    class FlagIterator,
    class OutputKeyIterator,
    class OutputValueIterator,
    class InequalityOp,
    class SelectedCountOutputIterator,
    class... UnaryPredicates
>
inline
cudaError_t partition_impl(void * temporary_storage,
                          size_t& storage_size,
                          KeyIterator keys_input,
                          ValueIterator values_input,
                          FlagIterator flags,
                          OutputKeyIterator keys_output,
                          OutputValueIterator values_output,
                          SelectedCountOutputIterator selected_count_output,
                          const size_t size,
                          InequalityOp inequality_op,
                          const cudaStream_t stream,
                          bool debug_synchronous,
                          UnaryPredicates... predicates)
{
    using offset_type = OffsetT;
    using key_type = typename std::iterator_traits<KeyIterator>::value_type;
    using value_type = typename std::iterator_traits<ValueIterator>::value_type;


    using config = default_or_custom_config<
        Config,
        default_select_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    using offset_scan_state_type = detail::lookback_scan_state<offset_type>;
    using offset_scan_state_with_sleep_type = detail::lookback_scan_state<offset_type, true>;
    using ordered_block_id_type = detail::ordered_block_id<unsigned int>;


    static constexpr unsigned int block_size = config::block_size;
    static constexpr unsigned int items_per_thread = config::items_per_thread;
    static constexpr auto items_per_block = block_size * items_per_thread;

    static constexpr bool is_three_way = sizeof...(UnaryPredicates) == 2;

    static constexpr size_t size_limit = config::size_limit;
    static constexpr size_t aligned_size_limit = ::rocprim::max<size_t>(size_limit - (size_limit % items_per_block), items_per_block);
    const size_t limited_size = std::min<size_t>(size, aligned_size_limit);
    const bool use_limited_size = limited_size == aligned_size_limit;

    const unsigned int number_of_blocks = 
        static_cast<unsigned int>(::rocprim::detail::ceiling_div(limited_size, items_per_block));


    size_t offset_scan_state_bytes = ::rocprim::detail::align_size(

        offset_scan_state_type::get_storage_size(number_of_blocks)
    );
    size_t ordered_block_id_bytes = ::rocprim::detail::align_size(
        ordered_block_id_type::get_storage_size(),
        alignof(size_t)
    );

    if(temporary_storage == nullptr)
    {

        storage_size = offset_scan_state_bytes + ordered_block_id_bytes + (sizeof(size_t) * 2 * (is_three_way ? 2 : 1));

        return cudaSuccess;
    }


    std::chrono::high_resolution_clock::time_point start;


    auto offset_scan_state = offset_scan_state_type::create(
        temporary_storage, number_of_blocks
    );
    auto offset_scan_state_with_sleep = offset_scan_state_with_sleep_type::create(
        temporary_storage, number_of_blocks
    );

    auto ptr = reinterpret_cast<char*>(temporary_storage);
    auto ordered_bid = ordered_block_id_type::create(
        reinterpret_cast<ordered_block_id_type::id_type*>(ptr + offset_scan_state_bytes)
    );

    size_t* selected_count = reinterpret_cast<size_t*>(ptr + offset_scan_state_bytes
                                                       + ordered_block_id_bytes);
    size_t* prev_selected_count
        = reinterpret_cast<size_t*>(ptr + offset_scan_state_bytes + ordered_block_id_bytes
                                    + (is_three_way ? 2 : 1) * sizeof(size_t));

    cudaError_t error;


    error = cudaMemsetAsync(selected_count,
                           0,
                           sizeof(*selected_count) * 2 * (is_three_way ? 2 : 1),
                           stream);
    if (error != cudaSuccess) return error;

    cudaDeviceProp prop;
    int deviceId;
    static_cast<void>(cudaGetDevice(&deviceId));
    static_cast<void>(cudaGetDeviceProperties(&prop, deviceId));


    int asicRevision = 0;


    const size_t number_of_launches = ::rocprim::detail::ceiling_div(size, aligned_size_limit);

    if(debug_synchronous)
    {
        std::cout << "use_limited_size " << use_limited_size << '\n';
        std::cout << "aligned_size_limit " << aligned_size_limit << '\n';
        std::cout << "number_of_launches " << number_of_launches << '\n';
        std::cout << "size " << size << '\n';
        std::cout << "block_size " << block_size << '\n';
        std::cout << "number of blocks " << number_of_blocks << '\n';
        std::cout << "items_per_block " << items_per_block << '\n';
    }

    for (size_t i = 0, offset = 0; i < number_of_launches; i++, offset+=limited_size)
    {
        const unsigned int current_size = static_cast<unsigned int>(std::min<size_t>(size - offset, limited_size));

        const unsigned int current_number_of_blocks = ::rocprim::detail::ceiling_div(current_size, items_per_block);

        auto grid_size = ::rocprim::detail::ceiling_div(number_of_blocks, block_size);

        if(debug_synchronous)
        {
            std::cout << "current size " << current_size << '\n';
            std::cout << "current number of blocks " << current_number_of_blocks << '\n';

            start = std::chrono::high_resolution_clock::now();
        }


        
            init_lookback_scan_state_kernel<offset_scan_state_type>
                <<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                offset_scan_state, current_number_of_blocks, ordered_bid
            );
        

        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_offset_scan_state_kernel", current_number_of_blocks, start)

        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();

        grid_size = current_number_of_blocks;
        

        
            partition_kernel<
                    SelectMethod, OnlySelected, config
                >
                <<<dim3(grid_size), dim3(block_size), 0, stream>>>(
                keys_input + offset, values_input + offset, flags + offset, keys_output, values_output, selected_count, prev_selected_count,
                current_size, inequality_op, offset_scan_state, current_number_of_blocks, ordered_bid, predicates...
            );
        

        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("partition_kernel", size, start)

        std::swap(selected_count, prev_selected_count);
    }

    error = ::rocprim::transform(
        prev_selected_count, selected_count_output, (is_three_way ? 2 : 1), 
        ::rocprim::identity<>{},
        stream, debug_synchronous
    );
    if (error != cudaSuccess) return error;

    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR
#undef ROCPRIM_DETAIL_HIP_SYNC

}
















































































template<
    class Config = default_config,
    class InputIterator,
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
cudaError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     FlagIterator flags,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     const cudaStream_t stream = 0,
                     const bool debug_synchronous = false)
{

    using unary_predicate_type = ::rocprim::empty_type;

    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    rocprim::empty_type* const no_values = nullptr;

    return detail::partition_impl<detail::select_method::flag, false, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous, unary_predicate_type()
    );
}
























































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate
>
inline
cudaError_t partition(void * temporary_storage,
                     size_t& storage_size,
                     InputIterator input,
                     OutputIterator output,
                     SelectedCountOutputIterator selected_count_output,
                     const size_t size,
                     UnaryPredicate predicate,
                     const cudaStream_t stream = 0,
                     const bool debug_synchronous = false)
{

    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;

    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    rocprim::empty_type* const no_values = nullptr;

    return detail::partition_impl<detail::select_method::predicate, false, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous, predicate
    );
}































































































































template <
    class Config = default_config,
    typename InputIterator,
    typename FirstOutputIterator,
    typename SecondOutputIterator,
    typename UnselectedOutputIterator,
    typename SelectedCountOutputIterator,
    typename FirstUnaryPredicate,
    typename SecondUnaryPredicate>
inline
cudaError_t partition_three_way(void * temporary_storage,
                               size_t& storage_size,
                               InputIterator input,
                               FirstOutputIterator output_first_part,
                               SecondOutputIterator output_second_part,
                               UnselectedOutputIterator output_unselected,
                               SelectedCountOutputIterator selected_count_output,
                               const size_t size,
                               FirstUnaryPredicate select_first_part_op,
                               SecondUnaryPredicate select_second_part_op,
                               const cudaStream_t stream = 0,
                               const bool debug_synchronous = false)
{

    using flag_type = ::rocprim::empty_type;
    flag_type * flags = nullptr;

    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = uint2;
    using output_key_iterator_tuple = tuple<
        FirstOutputIterator,
        SecondOutputIterator,
        UnselectedOutputIterator>;
    using output_value_iterator_tuple
        = tuple<::rocprim::empty_type*, ::rocprim::empty_type*, ::rocprim::empty_type*>;
    rocprim::empty_type* const no_input_values = nullptr;
    const output_value_iterator_tuple no_output_values {nullptr, nullptr, nullptr};

    output_key_iterator_tuple output{ output_first_part, output_second_part, output_unselected };

    return detail::partition_impl<detail::select_method::predicate, false, Config, offset_type>(
        temporary_storage, storage_size, input, no_input_values, flags, output, no_output_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous,
        select_first_part_op, select_second_part_op
    );
}




END_ROCPRIM_NAMESPACE

#endif
