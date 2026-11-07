



















#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_REDUCE_BY_KEY_HPP_

#include <iterator>
#include <utility>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../block/block_discontinuity.hpp"
#include "../../block/block_load_func.hpp"
#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<class Value>
struct carry_out
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    carry_out() = default;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    carry_out(const carry_out& rhs) = default;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    carry_out& operator=(const carry_out& rhs)
    {
        value = rhs.value;
        destination = rhs.destination;
        next_has_carry_in = rhs.next_has_carry_in;
        return *this;
    }

    Value value;
    unsigned int destination;
    bool next_has_carry_in;
};

template<class Value>
struct scan_by_key_pair
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    scan_by_key_pair() = default;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    scan_by_key_pair(const scan_by_key_pair& rhs) = default;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    scan_by_key_pair& operator=(const scan_by_key_pair& rhs)
    {
        key = rhs.key;
        value = rhs.value;
        return *this;
    }

    unsigned int key;
    Value value;
};





template<class Pair, class BinaryFunction>
struct scan_by_key_op
{
    BinaryFunction reduce_op;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    scan_by_key_op(BinaryFunction reduce_op)
        : reduce_op(reduce_op)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE
    Pair operator()(const Pair& a, const Pair& b)
    {
        Pair c;
        c.key = a.key + b.key;
        c.value = b.key != 0
            ? b.value
            : reduce_op(a.value, b.value);
        return c;
    }
};



template<class Key, class KeyCompareFunction>
struct key_flag_op
{
    KeyCompareFunction key_compare_op;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    key_flag_op(KeyCompareFunction key_compare_op)
        : key_compare_op(key_compare_op)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const Key& a, const Key& b)
    {
        return !key_compare_op(a, b);
    }
};



template<class Key, class KeyCompareFunction>
struct guarded_key_flag_op
{
    KeyCompareFunction key_compare_op;
    unsigned int valid_count;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    guarded_key_flag_op(KeyCompareFunction key_compare_op, unsigned int valid_count)
        : key_compare_op(key_compare_op), valid_count(valid_count)
    {}

    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const Key& a, const Key& b, unsigned int b_index)
    {
        return (b_index < valid_count && !key_compare_op(a, b)) || b_index == valid_count;
    }
};

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class KeyCompareFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void fill_unique_counts(KeysInputIterator keys_input,
                        unsigned int size,
                        unsigned int * unique_counts,
                        KeyCompareFunction key_compare_op,
                        unsigned int blocks_per_full_batch,
                        unsigned int full_batches)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;
    constexpr unsigned int warp_size = ::rocprim::device_warp_size();
    constexpr unsigned int warps_no = BlockSize / warp_size;

    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;

    using keys_load_type = ::rocprim::block_load<
        key_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using discontinuity_type = ::rocprim::block_discontinuity<key_type, BlockSize>;

    ROCPRIM_SHARED_MEMORY struct
    {
        union
        {
            typename keys_load_type::storage_type keys_load;
            typename discontinuity_type::storage_type discontinuity;
        };
        unsigned int unique_counts[warps_no];
    } storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int batch_id = ::rocprim::detail::block_id<0>();
    const unsigned int lane_id = ::rocprim::lane_id();
    const unsigned int warp_id = ::rocprim::warp_id<0, 1, 1>();

    unsigned int block_offset;
    unsigned int blocks_per_batch;
    if(batch_id < full_batches)
    {
        blocks_per_batch = blocks_per_full_batch;
        block_offset = batch_id * blocks_per_batch;
    }
    else
    {
        blocks_per_batch = blocks_per_full_batch - 1;
        block_offset = batch_id * blocks_per_batch + full_batches;
    }
    block_offset *= items_per_block;

    unsigned int warp_unique_count = 0;

    for(unsigned int bi = 0; bi < blocks_per_batch; bi++)
    {
        const bool is_last_block = (block_offset + items_per_block >= size);

        key_type keys[ItemsPerThread];
        unsigned int valid_count;
        ::rocprim::syncthreads();
        if(is_last_block)
        {
            valid_count = size - block_offset;
            keys_load_type().load(keys_input + block_offset, keys, valid_count, storage.keys_load);
        }
        else
        {
            valid_count = items_per_block;
            keys_load_type().load(keys_input + block_offset, keys, storage.keys_load);
        }

        bool tail_flags[ItemsPerThread];
        key_type successor_key = keys[ItemsPerThread - 1];
        ::rocprim::syncthreads();
        if(is_last_block)
        {
            discontinuity_type().flag_tails(
                tail_flags, successor_key, keys,
                guarded_key_flag_op<key_type, KeyCompareFunction>(key_compare_op, valid_count),
                storage.discontinuity
            );
        }
        else
        {
            if(flat_id == BlockSize - 1)
            {
                successor_key = keys_input[block_offset + items_per_block];
            }
            discontinuity_type().flag_tails(
                tail_flags, successor_key, keys,
                key_flag_op<key_type, KeyCompareFunction>(key_compare_op),
                storage.discontinuity
            );
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            warp_unique_count += ::rocprim::bit_count(::rocprim::ballot(tail_flags[i]));
        }

        block_offset += items_per_block;
    }

    if(lane_id == 0)
    {
        storage.unique_counts[warp_id] = warp_unique_count;
    }
    ::rocprim::syncthreads();

    if(flat_id == 0)
    {
        unsigned int batch_unique_count = 0;
        for(unsigned int w = 0; w < warps_no; w++)
        {
            batch_unique_count += storage.unique_counts[w];
        }
        unique_counts[batch_id] = batch_unique_count;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class UniqueCountOutputIterator
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void scan_unique_counts(unsigned int * unique_counts,
                        UniqueCountOutputIterator unique_count_output,
                        unsigned int batches)
{
    using load_type = ::rocprim::block_load<
        unsigned int, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using store_type = ::rocprim::block_store<
        unsigned int, BlockSize, ItemsPerThread,
        ::rocprim::block_store_method::block_store_transpose>;
    using scan_type = typename ::rocprim::block_scan<unsigned int, BlockSize>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename load_type::storage_type load;
        typename store_type::storage_type store;
        typename scan_type::storage_type scan;
    } storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

    unsigned int values[ItemsPerThread];
    load_type().load(unique_counts, values, batches, 0, storage.load);

    unsigned int unique_count;
    ::rocprim::syncthreads();
    scan_type().exclusive_scan(values, values, 0, unique_count);

    ::rocprim::syncthreads();
    store_type().store(unique_counts, values, batches, storage.store);

    if(flat_id == 0)
    {
        *unique_count_output = unique_count;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class KeysInputIterator,
    class ValuesInputIterator,
    class Result,
    class UniqueOutputIterator,
    class AggregatesOutputIterator,
    class KeyCompareFunction,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void reduce_by_key(KeysInputIterator keys_input,
                   ValuesInputIterator values_input,
                   unsigned int size,
                   const unsigned int * unique_starts,
                   carry_out<Result> * carry_outs,
                   Result * leading_aggregates,
                   UniqueOutputIterator unique_output,
                   AggregatesOutputIterator aggregates_output,
                   KeyCompareFunction key_compare_op,
                   BinaryFunction reduce_op,
                   unsigned int blocks_per_full_batch,
                   unsigned int full_batches)
{
    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    using key_type = typename std::iterator_traits<KeysInputIterator>::value_type;
    using result_type = Result;

    using keys_load_type = ::rocprim::block_load<
        key_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using values_load_type = ::rocprim::block_load<
        result_type, BlockSize, ItemsPerThread,
        ::rocprim::block_load_method::block_load_transpose>;
    using discontinuity_type = ::rocprim::block_discontinuity<key_type, BlockSize>;
    using scan_type = ::rocprim::block_scan<scan_by_key_pair<result_type>, BlockSize>;

    ROCPRIM_SHARED_MEMORY struct
    {
        union
        {
            typename keys_load_type::storage_type keys_load;
            typename values_load_type::storage_type values_load;
            typename discontinuity_type::storage_type discontinuity;
            typename scan_type::storage_type scan;
        };
        unsigned int unique_count;
        bool has_carry_in;
        detail::raw_storage<result_type> carry_in;
    } storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int batch_id = ::rocprim::detail::block_id<0>();

    unsigned int block_offset;
    unsigned int blocks_per_batch;
    if(batch_id < full_batches)
    {
        blocks_per_batch = blocks_per_full_batch;
        block_offset = batch_id * blocks_per_batch;
    }
    else
    {
        blocks_per_batch = blocks_per_full_batch - 1;
        block_offset = batch_id * blocks_per_batch + full_batches;
    }
    block_offset *= items_per_block;

    const unsigned int batch_start = unique_starts[batch_id];
    unsigned int block_start = batch_start;

    if(flat_id == 0)
    {
        storage.has_carry_in =
            (block_offset > 0) &&
            key_compare_op(keys_input[block_offset - 1], keys_input[block_offset]);
    }

    for(unsigned int bi = 0; bi < blocks_per_batch; bi++)
    {
        const bool is_last_block = (block_offset + items_per_block >= size);

        key_type keys[ItemsPerThread];
        result_type values[ItemsPerThread];
        unsigned int valid_count;
        if(is_last_block)
        {
            valid_count = size - block_offset;
            keys_load_type().load(keys_input + block_offset, keys, valid_count, storage.keys_load);
            ::rocprim::syncthreads();
            values_load_type().load(values_input + block_offset, values, valid_count, storage.values_load);
        }
        else
        {
            valid_count = items_per_block;
            keys_load_type().load(keys_input + block_offset, keys, storage.keys_load);
            ::rocprim::syncthreads();
            values_load_type().load(values_input + block_offset, values, storage.values_load);
        }

        if(bi > 0 && flat_id == 0 && storage.has_carry_in)
        {

            values[0] = reduce_op(storage.carry_in.get(), values[0]);
        }

        bool head_flags[ItemsPerThread];
        bool tail_flags[ItemsPerThread];
        key_type successor_key = keys[ItemsPerThread - 1];
        ::rocprim::syncthreads();
        if(is_last_block)
        {
            discontinuity_type().flag_heads_and_tails(
                head_flags, tail_flags, successor_key, keys,
                guarded_key_flag_op<key_type, KeyCompareFunction>(key_compare_op, valid_count),
                storage.discontinuity
            );
        }
        else
        {
            if(flat_id == BlockSize - 1)
            {
                successor_key = keys_input[block_offset + items_per_block];
            }
            discontinuity_type().flag_heads_and_tails(
                head_flags, tail_flags, successor_key, keys,
                key_flag_op<key_type, KeyCompareFunction>(key_compare_op),
                storage.discontinuity
            );
        }











        scan_by_key_pair<result_type> pairs[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            pairs[i].key = head_flags[i];
            pairs[i].value = values[i];
        }
        scan_by_key_op<scan_by_key_pair<result_type>, BinaryFunction> scan_op(reduce_op);
        ::rocprim::syncthreads();
        scan_type().inclusive_scan(pairs, pairs, storage.scan, scan_op);

        unsigned int ranks[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            ranks[i] = pairs[i].key - 1;
            values[i] = pairs[i].value;
        }

        if(flat_id == BlockSize - 1)
        {
            storage.unique_count = ranks[ItemsPerThread - 1] + (tail_flags[ItemsPerThread - 1] ? 1 : 0);
        }

        ::rocprim::syncthreads();
        const unsigned int unique_count = storage.unique_count;
        if(flat_id == 0)
        {


            head_flags[0] = !storage.has_carry_in;
        }
        if(is_last_block)
        {

            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                if(ranks[i] >= unique_count)
                {
                    head_flags[i] = false;
                }
            }
        }

        ::rocprim::syncthreads();
        if(flat_id == BlockSize - 1)
        {
            if(bi == blocks_per_batch - 1)
            {

                carry_outs[batch_id].value = values[ItemsPerThread - 1];
                carry_outs[batch_id].destination = block_start + ranks[ItemsPerThread - 1];
                carry_outs[batch_id].next_has_carry_in = !tail_flags[ItemsPerThread - 1];
            }
            else
            {

                storage.has_carry_in = !tail_flags[ItemsPerThread - 1];
                storage.carry_in.get() = values[ItemsPerThread - 1];
            }
        }
        if(batch_id > 0 && block_start == batch_start)
        {
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {




                if(tail_flags[i] && ranks[i] == 0)
                {
                    leading_aggregates[batch_id - 1] = values[i];
                }
            }
        }



        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(head_flags[i])
            {

                unique_output[block_start + ranks[i]] = keys[i];
            }
            if(tail_flags[i])
            {

                aggregates_output[block_start + ranks[i]] = values[i];
            }
        }

        block_offset += items_per_block;
        block_start += unique_count;
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    class Result,
    class AggregatesOutputIterator,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void scan_and_scatter_carry_outs(const carry_out<Result> * carry_outs,
                                 const Result * leading_aggregates,
                                 AggregatesOutputIterator aggregates_output,
                                 BinaryFunction reduce_op,
                                 unsigned int batches)
{
    using result_type = Result;

    using discontinuity_type = ::rocprim::block_discontinuity<unsigned int, BlockSize>;
    using scan_type = ::rocprim::block_scan<scan_by_key_pair<result_type>, BlockSize>;

    ROCPRIM_SHARED_MEMORY struct
    {
        typename discontinuity_type::storage_type discontinuity;
        typename scan_type::storage_type scan;
    } storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();

    carry_out<result_type> cs[ItemsPerThread];
    block_load_direct_blocked(flat_id, carry_outs, cs, batches - 1);

    unsigned int destinations[ItemsPerThread];
    result_type values[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        destinations[i] = cs[i].destination;
        values[i] = cs[i].value;
    }

    bool head_flags[ItemsPerThread];
    bool tail_flags[ItemsPerThread];
    ::rocprim::equal_to<unsigned int> compare_op;


    discontinuity_type().flag_heads_and_tails(
        head_flags, tail_flags,
        destinations[ItemsPerThread - 1],
        destinations,
        guarded_key_flag_op<unsigned int, decltype(compare_op)>(compare_op, batches - 1),
        storage.discontinuity
    );

    scan_by_key_pair<result_type> pairs[ItemsPerThread];
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        pairs[i].key = head_flags[i];
        pairs[i].value = values[i];
    }

    scan_by_key_op<scan_by_key_pair<result_type>, BinaryFunction> scan_op(reduce_op);
    scan_type().inclusive_scan(pairs, pairs, storage.scan, scan_op);


    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        if(tail_flags[i])
        {
            const unsigned int dst = destinations[i];
            const result_type aggregate = pairs[i].value;
            if(cs[i].next_has_carry_in)
            {


                aggregates_output[dst] = reduce_op(aggregate, leading_aggregates[flat_id * ItemsPerThread + i]);
            }
            else
            {

                aggregates_output[dst] = aggregate;
            }
        }
    }
}

}

END_ROCPRIM_NAMESPACE

#endif
