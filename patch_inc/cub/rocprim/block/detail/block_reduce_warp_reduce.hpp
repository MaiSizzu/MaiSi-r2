



















#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_WARP_REDUCE_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_REDUCE_WARP_REDUCE_HPP_

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
    unsigned int BlockSizeZ
>
class block_reduce_warp_reduce
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    static constexpr unsigned int warp_size_ =
        detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());

    static constexpr unsigned int warps_no_ = (BlockSize + warp_size_ - 1) / warp_size_;


    static constexpr bool block_size_is_warp_multiple_ = ((BlockSize % warp_size_) == 0);
    static constexpr bool warps_no_is_pow_of_two_ = detail::is_power_of_two(warps_no_);





    using warp_reduce_input_type = ::rocprim::detail::warp_reduce_crosslane<T, warp_size_, false>;


    using warp_reduce_output_type = ::rocprim::detail::warp_reduce_crosslane<
        T, detail::next_power_of_two(warps_no_), false
    >;

    struct storage_type_
    {
        T warp_partials[warps_no_];
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
    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce_impl(const unsigned int flat_tid,
                     T input,
                     T& output,
                     storage_type& storage,
                     BinaryFunction reduce_op)
    {
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        const auto lane_id = ::rocprim::lane_id();
        const unsigned int warp_offset = warp_id * warp_size_;
        const unsigned int num_valid =
            (warp_offset < BlockSize) ? BlockSize - warp_offset : 0;
        storage_type_& storage_ = storage.get();


        warp_reduce<!block_size_is_warp_multiple_, warp_reduce_input_type>(
            input, output, num_valid, reduce_op
        );


        if(lane_id == 0)
        {
            storage_.warp_partials[warp_id] = output;
        }
        ::rocprim::syncthreads();

        if(flat_tid < warps_no_)
        {

            auto warp_partial = storage_.warp_partials[lane_id];

            warp_reduce<!warps_no_is_pow_of_two_, warp_reduce_output_type>(
                warp_partial, output, warps_no_, reduce_op
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
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        const auto lane_id = ::rocprim::lane_id();
        const unsigned int warp_offset = warp_id * warp_size_;
        const unsigned int num_valid =
            (warp_offset < valid_items) ? valid_items - warp_offset : 0;
        storage_type_& storage_ = storage.get();


        warp_reduce_input_type().reduce(
            input, output, num_valid, reduce_op
        );


        if(lane_id == 0)
        {
            storage_.warp_partials[warp_id] = output;
        }
        ::rocprim::syncthreads();

        if(flat_tid < warps_no_)
        {

            auto warp_partial = storage_.warp_partials[lane_id];

            unsigned int valid_warps_no = (valid_items + warp_size_ - 1) / warp_size_;
            warp_reduce_output_type().reduce(
                warp_partial, output, valid_warps_no, reduce_op
            );
        }
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
