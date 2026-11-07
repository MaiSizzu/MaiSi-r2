



















#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_REDUCE_THEN_SCAN_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_REDUCE_THEN_SCAN_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_reduce.hpp"


BEGIN_ROCPRIM_NAMESPACE

namespace detail
{



template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto single_scan_block_scan(T (&input)[ItemsPerThread],
                            T (&output)[ItemsPerThread],
                            T initial_value,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op)
    -> typename std::enable_if<Exclusive>::type
{
    BlockScan()
        .exclusive_scan(
            input,
            output,
            initial_value,
            storage,
            scan_op
        );
}

template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto single_scan_block_scan(T (&input)[ItemsPerThread],
                            T (&output)[ItemsPerThread],
                            T initial_value,
                            typename BlockScan::storage_type& storage,
                            BinaryFunction scan_op)
    -> typename std::enable_if<!Exclusive>::type
{
    (void) initial_value;
    BlockScan()
        .inclusive_scan(
            input,
            output,
            storage,
            scan_op
        );
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void single_scan_kernel_impl(InputIterator input,
                             const size_t input_size,
                             ResultType initial_value,
                             OutputIterator output,
                             BinaryFunction scan_op)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

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

    result_type values[items_per_thread];

    block_load_type()
        .load(
            input,
            values,
            input_size,
            *(input),
            storage.load
        );
    ::rocprim::syncthreads();

    single_scan_block_scan<Exclusive, block_scan_type>(
        values,
        values,
        initial_value,
        storage.scan,
        scan_op
    );
    ::rocprim::syncthreads();


    block_store_type()
        .store(
            output,
            values,
            input_size,
            storage.store
        );
}



template<
    class Config,
    class InputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void block_reduce_kernel_impl(InputIterator input,
                              BinaryFunction scan_op,
                              ResultType * block_prefixes)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

    using result_type = ResultType;
    using block_reduce_type = ::rocprim::block_reduce<
        result_type, block_size,
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
    using block_load_type = ::rocprim::block_load<
        result_type, block_size, items_per_thread,
        Config::block_load_method
    >;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_type::storage_type load;
        typename block_reduce_type::storage_type reduce;
    } storage;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_thread * block_size;


    result_type values[items_per_thread];
    result_type block_prefix;

    block_load_type()
        .load(
            input + block_offset,
            values,
            storage.load
        );
    ::rocprim::syncthreads();

    block_reduce_type()
        .reduce(
            values,
            block_prefix,
            storage.reduce,
            scan_op
        );


    if(flat_id == 0)
    {
        block_prefixes[flat_block_id] = block_prefix;
    }
}



template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto final_scan_block_scan(const unsigned int flat_block_id,
                           T (&input)[ItemsPerThread],
                           T (&output)[ItemsPerThread],
                           T initial_value,
                           ResultType * block_prefixes,
                           typename BlockScan::storage_type& storage,
                           BinaryFunction scan_op)
    -> typename std::enable_if<Exclusive>::type
{
    if(flat_block_id != 0)
    {

        initial_value = scan_op(
            initial_value, block_prefixes[flat_block_id - 1]
        );
    }
    BlockScan()
        .exclusive_scan(
            input,
            output,
            initial_value,
            storage,
            scan_op
        );
}

template<
    bool Exclusive,
    class BlockScan,
    class T,
    unsigned int ItemsPerThread,
    class ResultType,
    class BinaryFunction
>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto final_scan_block_scan(const unsigned int flat_block_id,
                           T (&input)[ItemsPerThread],
                           T (&output)[ItemsPerThread],
                           T initial_value,
                           ResultType * block_prefixes,
                           typename BlockScan::storage_type& storage,
                           BinaryFunction scan_op)
    -> typename std::enable_if<!Exclusive>::type
{
    (void) initial_value;
    if(flat_block_id == 0)
    {
        BlockScan()
            .inclusive_scan(
                input,
                output,
                storage,
                scan_op
            );
    }
    else
    {
        auto block_prefix_op =
            [&block_prefixes, &flat_block_id](const T& )
            {
                return block_prefixes[flat_block_id - 1];
            };
        BlockScan()
            .inclusive_scan(
                input,
                output,
                storage,
                block_prefix_op,
                scan_op
            );
    }
}

template<
    bool Exclusive,
    class Config,
    class InputIterator,
    class OutputIterator,
    class BinaryFunction,
    class ResultType
>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
void final_scan_kernel_impl(InputIterator input,
                            const size_t input_size,
                            OutputIterator output,
                            ResultType initial_value,
                            BinaryFunction scan_op,
                            ResultType * block_prefixes,
                            ResultType * previous_last_element = nullptr,
                            ResultType * new_last_element = nullptr,
                            bool override_first_value = false,
                            bool save_last_value = false)
{
    constexpr unsigned int block_size = Config::block_size;
    constexpr unsigned int items_per_thread = Config::items_per_thread;

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


    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();

    constexpr unsigned int items_per_block = block_size * items_per_thread;
    const unsigned int block_offset = flat_block_id * items_per_block;

    const unsigned int number_of_blocks = (input_size + items_per_block - 1)/items_per_block;


    result_type values[items_per_thread];


    auto valid_in_last_block = input_size - items_per_block * (number_of_blocks - 1);

    if(flat_block_id == (number_of_blocks - 1))
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                valid_in_last_block,
                *(input + block_offset),
                storage.load
            );
    }
    else
    {
        block_load_type()
            .load(
                input + block_offset,
                values,
                storage.load
            );
    }
    ::rocprim::syncthreads();



    if(override_first_value && flat_block_id == 0)
    {
        if(Exclusive)
            initial_value = scan_op(previous_last_element[0], *(input-1));
        else if(::rocprim::detail::block_thread_id<0>() == 0)
            values[0] = scan_op(previous_last_element[0], values[0]);
    }

    final_scan_block_scan<Exclusive, block_scan_type>(
        flat_block_id,
        values,
        values,
        initial_value,
        block_prefixes,
        storage.scan,
        scan_op
    );
    ::rocprim::syncthreads();


    if(flat_block_id == (number_of_blocks - 1))
    {
        block_store_type()
            .store(
                output + block_offset,
                values,
                valid_in_last_block,
                storage.store
            );

        if(save_last_value &&
           (::rocprim::detail::block_thread_id<0>() ==
           (valid_in_last_block - 1) / items_per_thread))
        {
            for(unsigned int i = 0; i < items_per_thread; i++)
            {
                if(i == (valid_in_last_block - 1) % items_per_thread)
                {
                    new_last_element[0] = values[i];
                }
            }
        }
    }
    else
    {
        block_store_type()
            .store(
                output + block_offset,
                values,
                storage.store
            );
    }
}


template<class T>
size_t scan_get_temporary_storage_bytes(size_t input_size,
                                        size_t items_per_block)
{
    if(input_size <= items_per_block)
    {
        return 0;
    }
    auto size = (input_size + items_per_block - 1)/(items_per_block);
    return size * sizeof(T) + scan_get_temporary_storage_bytes<T>(size, items_per_block);
}

}

END_ROCPRIM_NAMESPACE

#endif
