



















#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_REDUCE_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SEGMENTED_REDUCE_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load_func.hpp"
#include "../../block/block_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class Config,
    class InputIterator,
    class OutputIterator,
    class OffsetIterator,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void segmented_reduce(InputIterator input,
                      OutputIterator output,
                      OffsetIterator begin_offsets,
                      OffsetIterator end_offsets,
                      BinaryFunction reduce_op,
                      ResultType initial_value)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;
    constexpr unsigned int items_per_block = block_size * items_per_thread;

    using reduce_type = ::rocprim::block_reduce<
        ResultType, block_size,
        Config::block_reduce_method
    >;

    ROCPRIM_SHARED_MEMORY typename reduce_type::storage_type reduce_storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int segment_id = ::rocprim::detail::block_id<0>();

    const unsigned int begin_offset = begin_offsets[segment_id];
    const unsigned int end_offset = end_offsets[segment_id];


    if(end_offset <= begin_offset)
    {
        if(flat_id == 0)
        {
            output[segment_id] = initial_value;
        }
        return;
    }

    ResultType result;
    unsigned int block_offset = begin_offset;
    if(block_offset + items_per_block > end_offset)
    {



        const unsigned int valid_count = end_offset - block_offset;
        if(flat_id < valid_count)
        {
            unsigned int offset = block_offset + flat_id;
            result = input[offset];
            offset += block_size;
            while(offset < end_offset)
            {
                result = reduce_op(result, static_cast<ResultType>(input[offset]));
                offset += block_size;
            }
        }


        if(valid_count >= block_size)
        {

            reduce_type().reduce(result, result, reduce_storage, reduce_op);
        }
        else
        {
            reduce_type().reduce(result, result, valid_count, reduce_storage, reduce_op);
        }
    }
    else
    {


        ResultType values[items_per_thread];


        block_load_direct_striped<block_size>(flat_id, input + block_offset, values);
        result = values[0];
        for(unsigned int i = 1; i < items_per_thread; i++)
        {
            result = reduce_op(result, values[i]);
        }
        block_offset += items_per_block;


        while(block_offset + items_per_block < end_offset)
        {
            block_load_direct_striped<block_size>(flat_id, input + block_offset, values);
            for(unsigned int i = 0; i < items_per_thread; i++)
            {
                result = reduce_op(result, values[i]);
            }
            block_offset += items_per_block;
        }


        const unsigned int valid_count = end_offset - block_offset;
        block_load_direct_striped<block_size>(flat_id, input + block_offset, values, valid_count);
        for(unsigned int i = 0; i < items_per_thread; i++)
        {
            if(i * block_size + flat_id < valid_count)
            {
                result = reduce_op(result, values[i]);
            }
        }


        reduce_type().reduce(result, result, reduce_storage, reduce_op);
    }

    if(flat_id == 0)
    {
        output[segment_id] = reduce_op(initial_value, result);
    }
}

}

END_ROCPRIM_NAMESPACE

#endif
