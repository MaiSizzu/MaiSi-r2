



















#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_HPP_

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

#include "../block/block_load.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/reverse_iterator.hpp"
#include "detail/device_segmented_radix_sort.hpp"
#include "device_partition.hpp"
#include "device_segmented_radix_sort_config.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Config,
    bool Descending,
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void segmented_sort_kernel(KeysInputIterator keys_input,
                           typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                           KeysOutputIterator keys_output,
                           ValuesInputIterator values_input,
                           typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                           ValuesOutputIterator values_output,
                           bool to_output,
                           OffsetIterator begin_offsets,
                           OffsetIterator end_offsets,
                           unsigned int long_iterations,
                           unsigned int short_iterations,
                           unsigned int begin_bit,
                           unsigned int end_bit)
{
    segmented_sort<Config, Descending>(
        keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
        to_output,
        begin_offsets, end_offsets,
        long_iterations, short_iterations,
        begin_bit, end_bit
    );
}

template<
    class Config,
    bool Descending,
    unsigned int BlockSize,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class SegmentIndexIterator,
    class OffsetIterator
>
ROCPRIM_KERNEL
__launch_bounds__(BlockSize)
void segmented_sort_large_kernel(KeysInputIterator keys_input,
                                 typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                 KeysOutputIterator keys_output,
                                 ValuesInputIterator values_input,
                                 typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                 ValuesOutputIterator values_output,
                                 bool to_output,
                                 SegmentIndexIterator segment_indices,
                                 OffsetIterator begin_offsets,
                                 OffsetIterator end_offsets,
                                 unsigned int long_iterations,
                                 unsigned int short_iterations,
                                 unsigned int begin_bit,
                                 unsigned int end_bit)
{
    segmented_sort_large<Config, Descending>(
        keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
        to_output, segment_indices,
        begin_offsets, end_offsets,
        long_iterations, short_iterations,
        begin_bit, end_bit
    );
}

template<class Config,
         bool         Descending,
         unsigned int BlockSize,
         class KeysInputIterator,
         class KeysOutputIterator,
         class ValuesInputIterator,
         class ValuesOutputIterator,
         class SegmentIndexIterator,
         class OffsetIterator>
ROCPRIM_KERNEL __launch_bounds__(BlockSize) void segmented_sort_small_or_medium_kernel(
    KeysInputIterator                                               keys_input,
    typename std::iterator_traits<KeysInputIterator>::value_type*   keys_tmp,
    KeysOutputIterator                                              keys_output,
    ValuesInputIterator                                             values_input,
    typename std::iterator_traits<ValuesInputIterator>::value_type* values_tmp,
    ValuesOutputIterator                                            values_output,
    bool                                                            to_output,
    unsigned int                                                    num_segments,
    SegmentIndexIterator                                            segment_indices,
    OffsetIterator                                                  begin_offsets,
    OffsetIterator                                                  end_offsets,
    unsigned int                                                    begin_bit,
    unsigned int                                                    end_bit)
{
    segmented_sort_small<Config, Descending>(
        keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
        to_output, num_segments, segment_indices,
        begin_offsets, end_offsets,
        begin_bit, end_bit
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

struct TwoWayPartitioner
{
    template<typename InputIterator,
             typename FirstOutputIterator,
             typename SecondOutputIterator,
             typename UnselectedOutputIterator,
             typename SelectedCountOutputIterator,
             typename FirstUnaryPredicate,
             typename SecondUnaryPredicate>
    cudaError_t operator()(void*               temporary_storage,
                          size_t&             storage_size,
                          InputIterator       input,
                          FirstOutputIterator output_first_part,
                          SecondOutputIterator ,
                          UnselectedOutputIterator ,
                          SelectedCountOutputIterator selected_count_output,
                          const size_t                size,
                          FirstUnaryPredicate         select_first_part_op,
                          SecondUnaryPredicate ,
                          const cudaStream_t stream,
                          const bool        debug_synchronous)
    {
        return partition(temporary_storage,
                         storage_size,
                         input,
                         output_first_part,
                         selected_count_output,
                         size,
                         select_first_part_op,
                         stream,
                         debug_synchronous);
    }
};

struct ThreeWayPartitioner
{
    template<typename InputIterator,
             typename FirstOutputIterator,
             typename SecondOutputIterator,
             typename UnselectedOutputIterator,
             typename SelectedCountOutputIterator,
             typename FirstUnaryPredicate,
             typename SecondUnaryPredicate>
    cudaError_t operator()(void*                       temporary_storage,
                          size_t&                     storage_size,
                          InputIterator               input,
                          FirstOutputIterator         output_first_part,
                          SecondOutputIterator        output_second_part,
                          UnselectedOutputIterator    output_unselected,
                          SelectedCountOutputIterator selected_count_output,
                          const size_t                size,
                          FirstUnaryPredicate         select_first_part_op,
                          SecondUnaryPredicate        select_second_part_op,
                          const cudaStream_t           stream,
                          const bool                  debug_synchronous)
    {
        return partition_three_way(temporary_storage,
                                   storage_size,
                                   input,
                                   output_first_part,
                                   output_second_part,
                                   output_unselected,
                                   selected_count_output,
                                   size,
                                   select_first_part_op,
                                   select_second_part_op,
                                   stream,
                                   debug_synchronous);
    }
};

template<
    class Config,
    bool Descending,
    class KeysInputIterator,
    class KeysOutputIterator,
    class ValuesInputIterator,
    class ValuesOutputIterator,
    class OffsetIterator
>
inline
cudaError_t segmented_radix_sort_impl(void * temporary_storage,
                                     size_t& storage_size,
                                     KeysInputIterator keys_input,
                                     typename std::iterator_traits<KeysInputIterator>::value_type * keys_tmp,
                                     KeysOutputIterator keys_output,
                                     ValuesInputIterator values_input,
                                     typename std::iterator_traits<ValuesInputIterator>::value_type * values_tmp,
                                     ValuesOutputIterator values_output,
                                     unsigned int size,
                                     bool& is_result_in_output,
                                     unsigned int segments,
                                     OffsetIterator begin_offsets,
                                     OffsetIterator end_offsets,
                                     unsigned int begin_bit,
                                     unsigned int end_bit,
                                     cudaStream_t stream,
                                     bool debug_synchronous)
{
    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
    using segment_index_type = unsigned int;
    using segment_index_iterator = counting_iterator<segment_index_type>;

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
        default_segmented_radix_sort_config<ROCPRIM_TARGET_ARCH, key_type, value_type>
    >;

    static constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;
    static constexpr bool partitioning_allowed =
        !std::is_same<typename config::warp_sort_config, DisabledWarpSortConfig>::value;
    static constexpr unsigned int max_small_segment_length
        = config::warp_sort_config::items_per_thread_small
          * config::warp_sort_config::logical_warp_size_small;
    static constexpr unsigned int small_segments_per_block
        = config::warp_sort_config::block_size_small
          / config::warp_sort_config::logical_warp_size_small;
    static constexpr unsigned int max_medium_segment_length
        = config::warp_sort_config::items_per_thread_medium
          * config::warp_sort_config::logical_warp_size_medium;
    static constexpr unsigned int medium_segments_per_block
        = config::warp_sort_config::block_size_medium
          / config::warp_sort_config::logical_warp_size_medium;
    static_assert(
        max_small_segment_length <= max_medium_segment_length,
        "The max length of small segments cannot be higher than the max length of medium segments");

    static constexpr bool three_way_partitioning
        = max_small_segment_length < max_medium_segment_length;
    using partitioner_type
        = std::conditional_t<three_way_partitioning, ThreeWayPartitioner, TwoWayPartitioner>;
    partitioner_type partitioner;

    const auto large_segment_selector = [=](const unsigned int segment_index) mutable -> bool
    {
        const unsigned int segment_length
            = end_offsets[segment_index] - begin_offsets[segment_index];
        return segment_length > max_medium_segment_length;
    };
    const auto medium_segment_selector = [=](const unsigned int segment_index) mutable -> bool
    {
        const unsigned int segment_length = end_offsets[segment_index] - begin_offsets[segment_index];
        return segment_length > max_small_segment_length;
    };

    const bool with_double_buffer = keys_tmp != nullptr;
    const unsigned int bits = end_bit - begin_bit;
    const unsigned int iterations = ::rocprim::detail::ceiling_div(bits, config::long_radix_bits);
    const bool to_output = with_double_buffer || (iterations - 1) % 2 == 0;
    is_result_in_output = (iterations % 2 == 0) != to_output;
    const unsigned int radix_bits_diff = config::long_radix_bits - config::short_radix_bits;
    const unsigned int short_iterations = radix_bits_diff != 0
        ? ::rocprim::min(iterations, (config::long_radix_bits * iterations - bits) / radix_bits_diff)
        : 0;
    const unsigned int long_iterations = iterations - short_iterations;
    const bool do_partitioning = partitioning_allowed
        && segments >= config::warp_sort_config::partitioning_threshold;

    const size_t keys_bytes = ::rocprim::detail::align_size(size * sizeof(key_type));
    const size_t values_bytes = with_values ? ::rocprim::detail::align_size(size * sizeof(value_type)) : 0;
    const size_t large_and_small_segment_indices_bytes
        = ::rocprim::detail::align_size(segments * sizeof(segment_index_type));
    const size_t medium_segment_indices_bytes
        = three_way_partitioning
              ? ::rocprim::detail::align_size(segments * sizeof(segment_index_type))
              : 0;
    static constexpr size_t segment_count_output_size = three_way_partitioning ? 2 : 1;
    const size_t            segment_count_output_bytes
        = ::rocprim::detail::align_size(segment_count_output_size * sizeof(segment_index_type));

    segment_index_type* large_segment_indices_output{};


    auto small_segment_indices_output
        = make_reverse_iterator(large_segment_indices_output + segments);
    segment_index_type* medium_segment_indices_output{};
    segment_index_type* segment_count_output{};
    size_t              partition_storage_size{};
    void*               partition_temporary_storage{};
    if(temporary_storage == nullptr)
    {
        storage_size = with_double_buffer ? 0 : (keys_bytes + values_bytes);
        if(do_partitioning)
        {
            storage_size += large_and_small_segment_indices_bytes;
            storage_size += medium_segment_indices_bytes;
            storage_size += segment_count_output_bytes;
            const auto partition_result = partitioner(partition_temporary_storage,
                                                      partition_storage_size,
                                                      segment_index_iterator{},
                                                      large_segment_indices_output,
                                                      medium_segment_indices_output,
                                                      small_segment_indices_output,
                                                      segment_count_output,
                                                      segments,
                                                      large_segment_selector,
                                                      medium_segment_selector,
                                                      stream,
                                                      debug_synchronous);
            if(cudaSuccess != partition_result)
            {
                return partition_result;
            }
            storage_size += partition_storage_size;
        }



        storage_size = storage_size == 0 ? 4 : storage_size;
        return cudaSuccess;
    }
    if(segments == 0u)
    {
        return cudaSuccess;
    }
    if(debug_synchronous)
    {
        std::cout << "begin_bit " << begin_bit << '\n';
        std::cout << "end_bit " << end_bit << '\n';
        std::cout << "bits " << bits << '\n';
        std::cout << "segments " << segments << '\n';
        std::cout << "radix_bits_diff " << radix_bits_diff << '\n';
        std::cout << "storage_size " << storage_size << '\n';
        std::cout << "iterations " << iterations << '\n';
        std::cout << "long_iterations " << long_iterations << '\n';
        std::cout << "short_iterations " << short_iterations << '\n';
        std::cout << "do_partitioning " << do_partitioning << '\n';
        std::cout << "config::sort::block_size: " << config::sort::block_size << '\n';
        std::cout << "config::sort::items_per_thread: " << config::sort::items_per_thread << '\n';
        cudaError_t error = cudaStreamSynchronize(stream);
        if(error != cudaSuccess) return error;
    }

    char* ptr = reinterpret_cast<char*>(temporary_storage);
    if(!with_double_buffer)
    {
        keys_tmp = reinterpret_cast<key_type*>(ptr);
        ptr += keys_bytes;
        values_tmp = with_values ? reinterpret_cast<value_type*>(ptr) : nullptr;
        ptr += values_bytes;
    }
    large_segment_indices_output = reinterpret_cast<segment_index_type*>(ptr);
    ptr += large_and_small_segment_indices_bytes;
    medium_segment_indices_output = reinterpret_cast<segment_index_type*>(ptr);
    ptr += medium_segment_indices_bytes;
    small_segment_indices_output = make_reverse_iterator(large_segment_indices_output + segments);
    segment_count_output         = reinterpret_cast<segment_index_type*>(ptr);
    ptr += segment_count_output_bytes;
    partition_temporary_storage = ptr;
    ptr += partition_storage_size;

    if(do_partitioning)
    {
        cudaError_t result = partitioner(partition_temporary_storage,
                                        partition_storage_size,
                                        segment_index_iterator{},
                                        large_segment_indices_output,
                                        medium_segment_indices_output,
                                        small_segment_indices_output,
                                        segment_count_output,
                                        segments,
                                        large_segment_selector,
                                        medium_segment_selector,
                                        stream,
                                        debug_synchronous);
        if(cudaSuccess != result)
        {
            return result;
        }
        segment_index_type segment_counts[segment_count_output_size]{};
        result = cudaMemcpyAsync(&segment_counts,
                                segment_count_output,
                                segment_count_output_bytes,
                                cudaMemcpyDeviceToHost,
                                stream);
        if(cudaSuccess != result)
        {
            return result;
        }
        result = cudaStreamSynchronize(stream);
        if(cudaSuccess != result)
        {
            return result;
        }
        const auto large_segment_count  = segment_counts[0];
        const auto medium_segment_count = three_way_partitioning ? segment_counts[1] : 0;
        const auto small_segment_count  = segments - large_segment_count - medium_segment_count;
        if(debug_synchronous)
        {
            std::cout << "large_segment_count " << large_segment_count << '\n';
            std::cout << "medium_segment_count " << medium_segment_count << '\n';
            std::cout << "small_segment_count " << small_segment_count << '\n';
        }
        if(large_segment_count > 0)
        {
            std::chrono::high_resolution_clock::time_point start;
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            segmented_sort_large_kernel<config, Descending, config::sort::block_size>
                <<<dim3(large_segment_count), dim3(config::sort::block_size), 0, stream>>>(
                keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
                to_output, large_segment_indices_output,
                begin_offsets, end_offsets,
                long_iterations, short_iterations,
                begin_bit, end_bit
            );
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_sort:large_segments",
                                                        large_segment_count,
                                                        start)
        }
        if(three_way_partitioning && medium_segment_count > 0)
        {
            const auto medium_segment_grid_size
                = ::rocprim::detail::ceiling_div(medium_segment_count, medium_segments_per_block);
            std::chrono::high_resolution_clock::time_point start;
            if(debug_synchronous)
                start = std::chrono::high_resolution_clock::now();
            
                    segmented_sort_small_or_medium_kernel<
                        select_warp_sort_helper_config_medium_t<typename config::warp_sort_config>,
                        Descending,
                        config::warp_sort_config::block_size_medium>
                <<<dim3(medium_segment_grid_size),
                dim3(config::warp_sort_config::block_size_medium),
                0,
                stream>>>(
                keys_input,
                keys_tmp,
                keys_output,
                values_input,
                values_tmp,
                values_output,
                is_result_in_output,
                medium_segment_count,
                medium_segment_indices_output,
                begin_offsets,
                end_offsets,
                begin_bit,
                end_bit);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_sort:medium_segments",
                                                        medium_segment_count,
                                                        start)
        }
        if(small_segment_count > 0)
        {
            const auto small_segment_grid_size = ::rocprim::detail::ceiling_div(small_segment_count,
                                                                                small_segments_per_block);
            std::chrono::high_resolution_clock::time_point start;
            if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
            
                    segmented_sort_small_or_medium_kernel<
                        select_warp_sort_helper_config_small_t<typename config::warp_sort_config>,
                        Descending,
                        config::warp_sort_config::block_size_small>
                <<<dim3(small_segment_grid_size),
                dim3(config::warp_sort_config::block_size_small),
                0,
                stream>>>(
                keys_input,
                keys_tmp,
                keys_output,
                values_input,
                values_tmp,
                values_output,
                is_result_in_output,
                small_segment_count,
                small_segment_indices_output,
                begin_offsets,
                end_offsets,
                begin_bit,
                end_bit);
            ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_sort:small_segments",
                                                        small_segment_count,
                                                        start)
        }
    }
    else
    {
        std::chrono::high_resolution_clock::time_point start;
        if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
        segmented_sort_kernel<config, Descending, config::sort::block_size>
            <<<dim3(segments), dim3(config::sort::block_size), 0, stream>>>(
            keys_input, keys_tmp, keys_output, values_input, values_tmp, values_output,
            to_output,
            begin_offsets, end_offsets,
            long_iterations, short_iterations,
            begin_bit, end_bit
        );
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("segmented_sort", segments, start)
    }
    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR

}

























































































template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t segmented_radix_sort_keys(void * temporary_storage,
                                     size_t& storage_size,
                                     KeysInputIterator keys_input,
                                     KeysOutputIterator keys_output,
                                     unsigned int size,
                                     unsigned int segments,
                                     OffsetIterator begin_offsets,
                                     OffsetIterator end_offsets,
                                     unsigned int begin_bit = 0,
                                     unsigned int end_bit = 8 * sizeof(Key),
                                     cudaStream_t stream = 0,
                                     bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool ignored;
    return detail::segmented_radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}

























































































template<
    class Config = default_config,
    class KeysInputIterator,
    class KeysOutputIterator,
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t segmented_radix_sort_keys_desc(void * temporary_storage,
                                          size_t& storage_size,
                                          KeysInputIterator keys_input,
                                          KeysOutputIterator keys_output,
                                          unsigned int size,
                                          unsigned int segments,
                                          OffsetIterator begin_offsets,
                                          OffsetIterator end_offsets,
                                          unsigned int begin_bit = 0,
                                          unsigned int end_bit = 8 * sizeof(Key),
                                          cudaStream_t stream = 0,
                                          bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool ignored;
    return detail::segmented_radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values, nullptr, values,
        size, ignored,
        segments, begin_offsets, end_offsets,
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
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t segmented_radix_sort_pairs(void * temporary_storage,
                                      size_t& storage_size,
                                      KeysInputIterator keys_input,
                                      KeysOutputIterator keys_output,
                                      ValuesInputIterator values_input,
                                      ValuesOutputIterator values_output,
                                      unsigned int size,
                                      unsigned int segments,
                                      OffsetIterator begin_offsets,
                                      OffsetIterator end_offsets,
                                      unsigned int begin_bit = 0,
                                      unsigned int end_bit = 8 * sizeof(Key),
                                      cudaStream_t stream = 0,
                                      bool debug_synchronous = false)
{
    bool ignored;
    return detail::segmented_radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        segments, begin_offsets, end_offsets,
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
    class OffsetIterator,
    class Key = typename std::iterator_traits<KeysInputIterator>::value_type
>
inline
cudaError_t segmented_radix_sort_pairs_desc(void * temporary_storage,
                                           size_t& storage_size,
                                           KeysInputIterator keys_input,
                                           KeysOutputIterator keys_output,
                                           ValuesInputIterator values_input,
                                           ValuesOutputIterator values_output,
                                           unsigned int size,
                                           unsigned int segments,
                                           OffsetIterator begin_offsets,
                                           OffsetIterator end_offsets,
                                           unsigned int begin_bit = 0,
                                           unsigned int end_bit = 8 * sizeof(Key),
                                           cudaStream_t stream = 0,
                                           bool debug_synchronous = false)
{
    bool ignored;
    return detail::segmented_radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys_input, nullptr, keys_output,
        values_input, nullptr, values_output,
        size, ignored,
        segments, begin_offsets, end_offsets,
        begin_bit, end_bit,
        stream, debug_synchronous
    );
}





























































































template<
    class Config = default_config,
    class Key,
    class OffsetIterator
>
inline
cudaError_t segmented_radix_sort_keys(void * temporary_storage,
                                     size_t& storage_size,
                                     double_buffer<Key>& keys,
                                     unsigned int size,
                                     unsigned int segments,
                                     OffsetIterator begin_offsets,
                                     OffsetIterator end_offsets,
                                     unsigned int begin_bit = 0,
                                     unsigned int end_bit = 8 * sizeof(Key),
                                     cudaStream_t stream = 0,
                                     bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool is_result_in_output;
    cudaError_t error = detail::segmented_radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
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
    class OffsetIterator
>
inline
cudaError_t segmented_radix_sort_keys_desc(void * temporary_storage,
                                          size_t& storage_size,
                                          double_buffer<Key>& keys,
                                          unsigned int size,
                                          unsigned int segments,
                                          OffsetIterator begin_offsets,
                                          OffsetIterator end_offsets,
                                          unsigned int begin_bit = 0,
                                          unsigned int end_bit = 8 * sizeof(Key),
                                          cudaStream_t stream = 0,
                                          bool debug_synchronous = false)
{
    empty_type * values = nullptr;
    bool is_result_in_output;
    cudaError_t error = detail::segmented_radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values, values, values,
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
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
    class OffsetIterator
>
inline
cudaError_t segmented_radix_sort_pairs(void * temporary_storage,
                                      size_t& storage_size,
                                      double_buffer<Key>& keys,
                                      double_buffer<Value>& values,
                                      unsigned int size,
                                      unsigned int segments,
                                      OffsetIterator begin_offsets,
                                      OffsetIterator end_offsets,
                                      unsigned int begin_bit = 0,
                                      unsigned int end_bit = 8 * sizeof(Key),
                                      cudaStream_t stream = 0,
                                      bool debug_synchronous = false)
{
    bool is_result_in_output;
    cudaError_t error = detail::segmented_radix_sort_impl<Config, false>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
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
    class OffsetIterator
>
inline
cudaError_t segmented_radix_sort_pairs_desc(void * temporary_storage,
                                           size_t& storage_size,
                                           double_buffer<Key>& keys,
                                           double_buffer<Value>& values,
                                           unsigned int size,
                                           unsigned int segments,
                                           OffsetIterator begin_offsets,
                                           OffsetIterator end_offsets,
                                           unsigned int begin_bit = 0,
                                           unsigned int end_bit = 8 * sizeof(Key),
                                           cudaStream_t stream = 0,
                                           bool debug_synchronous = false)
{
    bool is_result_in_output;
    cudaError_t error = detail::segmented_radix_sort_impl<Config, true>(
        temporary_storage, storage_size,
        keys.current(), keys.current(), keys.alternate(),
        values.current(), values.current(), values.alternate(),
        size, is_result_in_output,
        segments, begin_offsets, end_offsets,
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
