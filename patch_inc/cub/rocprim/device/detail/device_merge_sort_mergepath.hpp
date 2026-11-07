

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_MERGEPATH_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_MERGE_SORT_MERGEPATH_HPP_

#include <iterator>

#include "../../detail/various.hpp"

#include "device_merge_sort.hpp"
#include "device_merge.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

    template <unsigned int ItemsPerThread, class KeyT, class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void gmem_to_reg(KeyT (&output)[ItemsPerThread],
                     InputIterator input1,
                     InputIterator input2,
                     unsigned int count1,
                     unsigned int count2,
                     bool IsLastTile)
    {
        if(IsLastTile)
        {
            ROCPRIM_UNROLL
            for (unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                unsigned int idx = rocprim::flat_block_size() * item + threadIdx.x;
                if (idx < count1 + count2)
                {
                    output[item] = (idx < count1) ? input1[idx] : input2[idx - count1];
                }
            }

        }
        else
        {
            ROCPRIM_UNROLL
            for (unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                unsigned int idx = rocprim::flat_block_size() * item + threadIdx.x;
                output[item] = (idx < count1) ? input1[idx] : input2[idx - count1];
            }
        }
    }

    template <unsigned int BlockSize, unsigned int ItemsPerThread, class KeyT, class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reg_to_shared(OutputIterator output,
                       KeyT (&input)[ItemsPerThread])
    {
        ROCPRIM_UNROLL
        for (unsigned int item = 0; item < ItemsPerThread; ++item)
        {
            unsigned int idx = BlockSize * item + threadIdx.x;
            output[idx] = input[item];
        }
    }

    template<unsigned int BlockSize,
             unsigned int ItemsPerThread,
             class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator,
             class OffsetT,
             class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto
        block_merge_process_tile(KeysInputIterator    keys_input,
                                 KeysOutputIterator   keys_output,
                                 ValuesInputIterator  values_input,
                                 ValuesOutputIterator values_output,
                                 const OffsetT        input_size,
                                 const OffsetT        sorted_block_size,
                                 BinaryFunction       compare_function,
                                 const OffsetT*       merge_partitions)
            -> std::enable_if_t<
                (!std::is_trivially_copyable<
                     typename std::iterator_traits<ValuesInputIterator>::value_type>::value
                 || rocprim::is_floating_point<
                     typename std::iterator_traits<ValuesInputIterator>::value_type>::value
                 || std::is_integral<
                     typename std::iterator_traits<ValuesInputIterator>::value_type>::value),
                void>
    {
        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
        using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
        constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;
        constexpr unsigned int items_per_tile = BlockSize * ItemsPerThread;

        using block_store = block_store_impl<with_values, BlockSize, ItemsPerThread, key_type, value_type>;

        using keys_storage_ = key_type[items_per_tile + 1];
        using values_storage_ = value_type[items_per_tile + 1];

        ROCPRIM_SHARED_MEMORY union {
            typename block_store::storage_type   store;
            detail::raw_storage<keys_storage_>   keys;
            detail::raw_storage<values_storage_> values;
        } storage;

        auto& keys_shared = storage.keys.get();
        auto& values_shared = storage.values.get();

        const unsigned short flat_id = block_thread_id<0>();
        const unsigned int flat_block_id = block_id<0>();
        const bool IsIncompleteTile = flat_block_id == (input_size/items_per_tile);

        const OffsetT partition_beg = merge_partitions[flat_block_id];
        const OffsetT partition_end = merge_partitions[flat_block_id + 1];

        const unsigned int merged_tiles_number = sorted_block_size / items_per_tile;
        const unsigned int target_merged_tiles_number = merged_tiles_number * 2;
        const unsigned int mask  = target_merged_tiles_number - 1;
        const unsigned int tilegroup_start_id  = ~mask & flat_block_id;
        const OffsetT tilegroup_start = items_per_tile * tilegroup_start_id;

        const OffsetT diag = items_per_tile * flat_block_id - tilegroup_start;

        const OffsetT keys1_beg = partition_beg;
        OffsetT keys1_end = partition_end;
        const OffsetT keys2_beg = rocprim::min(input_size, 2 * tilegroup_start + sorted_block_size + diag - partition_beg);
        OffsetT keys2_end = rocprim::min(input_size, 2 * tilegroup_start + sorted_block_size + diag + items_per_tile - partition_end);

        if (mask == (mask & flat_block_id))
        {
            keys1_end = rocprim::min(input_size, tilegroup_start + sorted_block_size);
            keys2_end = rocprim::min(input_size, tilegroup_start + sorted_block_size * 2);
        }


        const unsigned int num_keys1 = static_cast<unsigned int>(keys1_end - keys1_beg);
        const unsigned int num_keys2 = static_cast<unsigned int>(keys2_end - keys2_beg);

        key_type keys[ItemsPerThread];
        gmem_to_reg<ItemsPerThread>(keys,
                                    keys_input + keys1_beg,
                                    keys_input + keys2_beg,
                                    num_keys1,
                                    num_keys2,
                                    IsIncompleteTile);

        reg_to_shared<BlockSize, ItemsPerThread>(keys_shared, keys);

        value_type values[ItemsPerThread];
        if ROCPRIM_IF_CONSTEXPR(with_values){
            gmem_to_reg<ItemsPerThread>(values,
                                        values_input + keys1_beg,
                                        values_input + keys2_beg,
                                        num_keys1,
                                        num_keys2,
                                        IsIncompleteTile);
        }
        rocprim::syncthreads();

        const unsigned int diag0_local = rocprim::min(num_keys1 + num_keys2, ItemsPerThread * flat_id);

        const unsigned int keys1_beg_local = merge_path(keys_shared,
                                                        &keys_shared[num_keys1],
                                                        num_keys1,
                                                        num_keys2,
                                                        diag0_local,
                                                        compare_function);
        const unsigned int keys1_end_local = num_keys1;
        const unsigned int keys2_beg_local = diag0_local - keys1_beg_local;
        const unsigned int keys2_end_local = num_keys2;
        range_t range_local = {keys1_beg_local,
                               keys1_end_local,
                               keys2_beg_local + keys1_end_local,
                               keys2_end_local + keys1_end_local};

        unsigned int indices[ItemsPerThread];

        serial_merge(keys_shared,
                     keys,
                     indices,
                     range_local,
                     compare_function);

        if ROCPRIM_IF_CONSTEXPR(with_values){
            reg_to_shared<BlockSize, ItemsPerThread>(values_shared, values);

            rocprim::syncthreads();

            ROCPRIM_UNROLL
            for (unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                values[item] = values_shared[indices[item]];
            }

            rocprim::syncthreads();
        }

        const OffsetT offset = flat_block_id * items_per_tile;
        block_store().store(offset,
                            input_size - offset,
                            IsIncompleteTile,
                            keys_output,
                            values_output,
                            keys,
                            values,
                            storage.store);
    }

    template<unsigned int BlockSize,
             unsigned int ItemsPerThread,
             class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator,
             class OffsetT,
             class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE auto
        block_merge_process_tile(KeysInputIterator    keys_input,
                                 KeysOutputIterator   keys_output,
                                 ValuesInputIterator  values_input,
                                 ValuesOutputIterator values_output,
                                 const OffsetT        input_size,
                                 const OffsetT        sorted_block_size,
                                 BinaryFunction       compare_function,
                                 const OffsetT*       merge_partitions)
            -> std::enable_if_t<
                (std::is_trivially_copyable<
                     typename std::iterator_traits<ValuesInputIterator>::value_type>::value
                 && !rocprim::is_floating_point<
                     typename std::iterator_traits<ValuesInputIterator>::value_type>::value
                 && !std::is_integral<
                     typename std::iterator_traits<ValuesInputIterator>::value_type>::value),
                void>
    {
        using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
        using value_type = typename std::iterator_traits<ValuesInputIterator>::value_type;
        constexpr bool with_values = !std::is_same<value_type, ::rocprim::empty_type>::value;
        constexpr unsigned int items_per_tile = BlockSize * ItemsPerThread;

        using block_store = block_store_impl<false, BlockSize, ItemsPerThread, key_type, value_type>;

        using keys_storage_ = key_type[items_per_tile + 1];
        using values_storage_ = value_type[items_per_tile + 1];

        ROCPRIM_SHARED_MEMORY union {
            typename block_store::storage_type   store;
            detail::raw_storage<keys_storage_>   keys;
            detail::raw_storage<values_storage_> values;
        } storage;

        auto& keys_shared = storage.keys.get();
        auto& values_shared = storage.values.get();

        const unsigned short flat_id = block_thread_id<0>();
        const unsigned int flat_block_id = block_id<0>();
        const bool IsIncompleteTile = flat_block_id == (input_size / items_per_tile);

        const OffsetT partition_beg = merge_partitions[flat_block_id];
        const OffsetT partition_end = merge_partitions[flat_block_id + 1];

        const unsigned int merged_tiles_number = sorted_block_size / items_per_tile;
        const unsigned int target_merged_tiles_number = merged_tiles_number * 2;
        const unsigned int mask  = target_merged_tiles_number - 1;
        const unsigned int tilegroup_start_id  = ~mask & flat_block_id;
        const OffsetT tilegroup_start = items_per_tile * tilegroup_start_id;

        const OffsetT diag = items_per_tile * flat_block_id - tilegroup_start;

        const OffsetT keys1_beg = partition_beg;
        OffsetT keys1_end = partition_end;
        const OffsetT keys2_beg = rocprim::min(input_size, 2 * tilegroup_start + sorted_block_size + diag - partition_beg);
        OffsetT keys2_end = rocprim::min(input_size, 2 * tilegroup_start + sorted_block_size + diag + items_per_tile - partition_end);

        if (mask == (mask & flat_block_id))
        {
            keys1_end = rocprim::min(input_size, tilegroup_start + sorted_block_size);
            keys2_end = rocprim::min(input_size, tilegroup_start + sorted_block_size * 2);
        }


        const unsigned int num_keys1 = static_cast<unsigned int>(keys1_end - keys1_beg);
        const unsigned int num_keys2 = static_cast<unsigned int>(keys2_end - keys2_beg);

        key_type keys[ItemsPerThread];
        gmem_to_reg<ItemsPerThread>(keys,
                                    keys_input + keys1_beg,
                                    keys_input + keys2_beg,
                                    num_keys1,
                                    num_keys2,
                                    IsIncompleteTile);

        reg_to_shared<BlockSize, ItemsPerThread>(keys_shared, keys);

        rocprim::syncthreads();

        const unsigned int diag0_local = rocprim::min(num_keys1 + num_keys2, ItemsPerThread * flat_id);

        const unsigned int keys1_beg_local = merge_path(keys_shared,
                                                        &keys_shared[num_keys1],
                                                        num_keys1,
                                                        num_keys2,
                                                        diag0_local,
                                                        compare_function);
        const unsigned int keys1_end_local = num_keys1;
        const unsigned int keys2_beg_local = diag0_local - keys1_beg_local;
        const unsigned int keys2_end_local = num_keys2;
        range_t range_local = {keys1_beg_local,
                               keys1_end_local,
                               keys2_beg_local + keys1_end_local,
                               keys2_end_local + keys1_end_local};

        unsigned int indices[ItemsPerThread];

        serial_merge(keys_shared,
                     keys,
                     indices,
                     range_local,
                     compare_function);

        if ROCPRIM_IF_CONSTEXPR(with_values)
        {
            const ValuesInputIterator input1 = values_input + keys1_beg;
            const ValuesInputIterator input2 = values_input + keys2_beg;
            if(IsIncompleteTile)
            {
                ROCPRIM_UNROLL
                for (unsigned int item = 0; item < ItemsPerThread; ++item)
                {
                    unsigned int idx = BlockSize * item + threadIdx.x;
                    if(idx < num_keys1)
                    {
                        values_shared[idx] = input1[idx];
                    }
                    else if(idx - num_keys1 < num_keys2)
                    {
                        values_shared[idx] = input2[idx - num_keys1];
                    }
                }
            }
            else
            {
                ROCPRIM_UNROLL
                for (unsigned int item = 0; item < ItemsPerThread; ++item)
                {
                    unsigned int idx = BlockSize * item + threadIdx.x;
                    if(idx < num_keys1)
                    {
                        values_shared[idx] = input1[idx];
                    }
                    else
                    {
                        values_shared[idx] = input2[idx - num_keys1];
                    }
                }
            }

            rocprim::syncthreads();

            const OffsetT offset = (flat_block_id * items_per_tile) + (threadIdx.x * ItemsPerThread);
            ROCPRIM_UNROLL
            for (unsigned int item = 0; item < ItemsPerThread; ++item)
            {
                values_output[offset + item] = values_shared[indices[item]];
            }

            rocprim::syncthreads();
        }

        const OffsetT offset = flat_block_id * items_per_tile;
        value_type values[ItemsPerThread];
        block_store().store(offset,
                           input_size - offset,
                           IsIncompleteTile,
                           keys_output,
                           values_output,
                           keys,
                           values,
                           storage.store);
    }

    template<unsigned int BlockSize,
             unsigned int ItemsPerThread,
             class KeysInputIterator,
             class KeysOutputIterator,
             class ValuesInputIterator,
             class ValuesOutputIterator,
             class OffsetT,
             class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
        block_merge_kernel_impl(KeysInputIterator    keys_input,
                                KeysOutputIterator   keys_output,
                                ValuesInputIterator  values_input,
                                ValuesOutputIterator values_output,
                                const OffsetT        input_size,
                                const OffsetT        sorted_block_size,
                                BinaryFunction       compare_function,
                                const OffsetT*       merge_partitions)
    {
        block_merge_process_tile<BlockSize, ItemsPerThread>(keys_input,
                                                            keys_output,
                                                            values_input,
                                                            values_output,
                                                            input_size,
                                                            sorted_block_size,
                                                            compare_function,
                                                            merge_partitions);
    }

}

END_ROCPRIM_NAMESPACE

#endif