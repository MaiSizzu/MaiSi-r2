



















#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_SCAN_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../detail/various.hpp"
#include "../../detail/binary_op_wrappers.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    bool Exclusive,
    bool UsePrefix,
    class BlockScanType,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto segmented_scan_block_scan(T (&input)[ItemsPerThread],
                               T (&output)[ItemsPerThread],
                               T& prefix,
                               typename BlockScanType::storage_type& storage,
                               BinaryFunction scan_op)
    -> typename std::enable_if<Exclusive>::type
{
    auto prefix_op =
        [&prefix, scan_op](const T& reduction)
        {
            auto saved_prefix = prefix;
            prefix = scan_op(prefix, reduction);
            return saved_prefix;
        };
    BlockScanType()
        .exclusive_scan(
            input, output,
            storage, prefix_op, scan_op
        );
}

template<
    bool Exclusive,
    bool UsePrefix,
    class BlockScanType,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto segmented_scan_block_scan(T (&input)[ItemsPerThread],
                               T (&output)[ItemsPerThread],
                               T& prefix,
                               typename BlockScanType::storage_type& storage,
                               BinaryFunction scan_op)
    -> typename std::enable_if<!Exclusive>::type
{
    if(UsePrefix)
    {
        auto prefix_op =
            [&prefix, scan_op](const T& reduction)
            {
                auto saved_prefix = prefix;
                prefix = scan_op(prefix, reduction);
                return saved_prefix;
            };
        BlockScanType()
            .inclusive_scan(
                input, output,
                storage, prefix_op, scan_op
            );
        return;
    }
    BlockScanType()
        .inclusive_scan(
            input, output, prefix,
            storage, scan_op
        );
}

template<
    bool Exclusive,
    class Config,
    class ResultType,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class InitValueType,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void segmented_scan(InputIterator input,
                    OutputIterator output,
                    OffsetIterator begin_offsets,
                    OffsetIterator end_offsets,
                    InitValueType initial_value,
                    BinaryFunction scan_op)
{
    constexpr auto block_size = Config::block_size;
    constexpr auto items_per_thread = Config::items_per_thread;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

    using result_type = ResultType;
    using block_load_type = ::rocprim::block_load<
        result_type, block_size, items_per_thread,
        Config::block_load_method
    >;
    using block_store_type = ::rocprim::block_store<
        result_type, block_size, items_per_thread,
        Config::block_store_method
    >;
    using block_scan_type = ::rocprim::block_scan<
        result_type, block_size,
        Config::block_scan_method
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_store_type::storage_type store;
        typename block_scan_type::storage_type scan;
    } storage;

    const unsigned int segment_id = ::rocprim::detail::block_id<0>();
    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset = end_offsets[segment_id];


    if(end_offset <= begin_offset)
    {
        return;
    }


    result_type values[items_per_thread];
    result_type prefix = initial_value;

    unsigned int block_offset = begin_offset;
    if(block_offset + items_per_block > end_offset)
    {



        const unsigned int valid_count = end_offset - block_offset;
        block_load_type().load(input + block_offset, values, valid_count, storage.load);
        ::rocprim::syncthreads();

        segmented_scan_block_scan<Exclusive, false, block_scan_type>(
            values, values, prefix, storage.scan, scan_op
        );
        ::rocprim::syncthreads();

        block_store_type().store(output + block_offset, values, valid_count, storage.store);
    }
    else
    {



        block_load_type().load(input + block_offset, values, storage.load);
        ::rocprim::syncthreads();

        segmented_scan_block_scan<Exclusive, false, block_scan_type>(
            values, values, prefix, storage.scan, scan_op
        );
        ::rocprim::syncthreads();

        block_store_type().store(output + block_offset, values, storage.store);
        ::rocprim::syncthreads();
        block_offset += items_per_block;


        while(block_offset + items_per_block < end_offset)
        {
            block_load_type().load(input + block_offset, values, storage.load);
            ::rocprim::syncthreads();

            segmented_scan_block_scan<Exclusive, true, block_scan_type>(
                values, values, prefix, storage.scan, scan_op
            );
            ::rocprim::syncthreads();
            block_store_type().store(output + block_offset, values, storage.store);
            ::rocprim::syncthreads();
            block_offset += items_per_block;
        }


        const unsigned int valid_count = end_offset - block_offset;
        block_load_type().load(input + block_offset, values, valid_count, storage.load);
        ::rocprim::syncthreads();

        segmented_scan_block_scan<Exclusive, true, block_scan_type>(
            values, values, prefix, storage.scan, scan_op
        );
        ::rocprim::syncthreads();

        block_store_type().store(output + block_offset, values, valid_count, storage.store);
    }
}

}

END_ROCPRIM_NAMESPACE

#endif
