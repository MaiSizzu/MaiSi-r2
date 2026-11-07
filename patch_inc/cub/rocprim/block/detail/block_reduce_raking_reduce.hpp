



















#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_RAKING_REDUCE_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_reduce.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ,
    bool CommutativeOnly = false
>
class block_reduce_raking_reduce
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    static constexpr unsigned int thread_reduction_size_ =
        (BlockSize + ::rocprim::device_warp_size() - 1)/ ::rocprim::device_warp_size();



    static constexpr unsigned int warp_size_ =
        detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());

    static constexpr bool commutative_only_        = CommutativeOnly && ((BlockSize % warp_size_ == 0) && (BlockSize > warp_size_));
    static constexpr unsigned int sharing_threads_ = ::rocprim::max<int>(1, BlockSize - warp_size_);
    static constexpr unsigned int segment_length_  = sharing_threads_ / warp_size_;


    static constexpr bool block_size_smaller_than_warp_size_ = (BlockSize < warp_size_);
    using warp_reduce_prefix_type = ::rocprim::detail::warp_reduce_crosslane<T, warp_size_, false>;

    struct storage_type_
    {
        T threads[BlockSize];
    };

public:
    using storage_type = detail::raw_storage<storage_type_>;






    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op)
    {
        this->reduce_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, storage, reduce_op
        );
    }





    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void reduce(T input,
                T& output,
                BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }






    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op)
    {

        T thread_input = input[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = reduce_op(thread_input, input[i]);
        }


        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->reduce_impl(
            flat_tid,
            thread_input, output,
            storage,
            reduce_op
        );
    }





    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, storage, reduce_op);
    }







    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                storage_type& storage,
                BinaryFunction reduce_op)
    {
        this->reduce_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, valid_items, storage, reduce_op
        );
    }







    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                BinaryFunction reduce_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->reduce(input, output, valid_items, storage, reduce_op);
    }

private:

    template<class BinaryFunction, bool FunctionCommutativeOnly = commutative_only_>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     storage_type& storage,
                     BinaryFunction reduce_op)
        -> typename std::enable_if<(!FunctionCommutativeOnly), void>::type
    {
        storage_type_& storage_ = storage.get();
        storage_.threads[flat_tid] = input;
        ::rocprim::syncthreads();

        if (flat_tid < warp_size_)
        {
            T thread_reduction = storage_.threads[flat_tid];
            for(unsigned int i = warp_size_ + flat_tid; i < BlockSize; i += warp_size_)
            {
                thread_reduction = reduce_op(
                    thread_reduction, storage_.threads[i]
                );
            }
            warp_reduce<block_size_smaller_than_warp_size_, warp_reduce_prefix_type>(
                thread_reduction, output, BlockSize, reduce_op
            );
        }
    }

    template<class BinaryFunction, bool FunctionCommutativeOnly = commutative_only_>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     storage_type& storage,
                     BinaryFunction reduce_op)
        -> typename std::enable_if<(FunctionCommutativeOnly), void>::type
    {
        storage_type_& storage_ = storage.get();

        if (flat_tid >= warp_size_)
            storage_.threads[flat_tid - warp_size_] = input;

        ::rocprim::syncthreads();

        if (flat_tid < warp_size_)
        {
            T thread_reduction = input;
            T* storage_pointer = &storage_.threads[flat_tid * segment_length_];
            #pragma unroll
            for( unsigned int i = 0; i < segment_length_; i++ )
            {
                thread_reduction = reduce_op(
                    thread_reduction, storage_pointer[i]
                );
            }
            warp_reduce<block_size_smaller_than_warp_size_, warp_reduce_prefix_type>(
                thread_reduction, output, BlockSize, reduce_op
            );
        }
    }

    template<bool UseValid, class WarpReduce, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto warp_reduce(T input,
                     T& output,
                     const unsigned int valid_items,
                     BinaryFunction reduce_op)
        -> typename std::enable_if<UseValid>::type
    {
        WarpReduce().reduce(
            input, output, valid_items, reduce_op
        );
    }

    template<bool UseValid, class WarpReduce, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto warp_reduce(T input,
                     T& output,
                     const unsigned int valid_items,
                     BinaryFunction reduce_op)
        -> typename std::enable_if<!UseValid>::type
    {
        (void) valid_items;
        WarpReduce().reduce(
            input, output, reduce_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     const unsigned int valid_items,
                     storage_type& storage,
                     BinaryFunction reduce_op)
    {
        storage_type_& storage_ = storage.get();
        storage_.threads[flat_tid] = input;
        ::rocprim::syncthreads();

        if (flat_tid < warp_size_)
        {
            T thread_reduction = storage_.threads[flat_tid];
            for(unsigned int i = warp_size_ + flat_tid; i < BlockSize; i += warp_size_)
            {
                if(i < valid_items)
                {
                    thread_reduction = reduce_op(thread_reduction, storage_.threads[i]);
                }
            }
            warp_reduce_prefix_type().reduce(thread_reduction, output, valid_items, reduce_op);
        }
    }
};
}

END_ROCPRIM_NAMESPACE

#endif
