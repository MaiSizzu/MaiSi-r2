



















#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_SCAN_WARP_SCAN_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../../warp/warp_scan.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_scan_warp_scan
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    static constexpr unsigned int warp_size_ =
        detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());

    static constexpr unsigned int warps_no_ = (BlockSize + warp_size_ - 1) / warp_size_;





    using warp_scan_input_type = ::rocprim::detail::warp_scan_crosslane<T, warp_size_>;


    using warp_scan_prefix_type = ::rocprim::detail::warp_scan_crosslane<T, detail::next_power_of_two(warps_no_)>;

    struct storage_type_
    {
        T warp_prefixes[warps_no_];











    };

public:
    using storage_type = detail::raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        this->inclusive_scan_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, storage, scan_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T input,
                        T& output,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->inclusive_scan(input, output, storage, scan_op);

        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, reduction, storage, scan_op);
    }

    template<class PrefixCallback, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        storage_type_& storage_ = storage.get();
        this->inclusive_scan_impl(flat_tid, input, output, storage, scan_op);

        T block_prefix = this->get_block_prefix(
            flat_tid, warp_id,
            storage_.warp_prefixes[warps_no_ - 1],
            prefix_callback_op, storage
        );
        output = scan_op(block_prefix, output);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        BinaryFunction scan_op)
    {

        T thread_input = input[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }


        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input,
            storage,
            scan_op
        );


        output[0] = input[0];
        if(flat_tid != 0)
        {
            output[0] = scan_op(thread_input, input[0]);
        }


        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            output[i] = scan_op(output[i-1], input[i]);
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, storage, scan_op);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->inclusive_scan(input, output, storage, scan_op);

        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->inclusive_scan(input, output, reduction, storage, scan_op);
    }

    template<
        class PrefixCallback,
        unsigned int ItemsPerThread,
        class BinaryFunction
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();

        T thread_input = input[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }


        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input,
            storage,
            scan_op
        );


        T block_prefix = this->get_block_prefix(
            flat_tid, ::rocprim::warp_id(flat_tid),
            storage_.warp_prefixes[warps_no_ - 1],
            prefix_callback_op, storage
        );


        output[0] = input[0];
        if(flat_tid != 0)
        {
            output[0] = scan_op(thread_input, input[0]);
        }

        output[0] = scan_op(block_prefix, output[0]);

        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            output[i] = scan_op(output[i-1], input[i]);
        }
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        this->exclusive_scan_impl(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, init, storage, scan_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(
            input, output, init, storage, scan_op
        );
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->exclusive_scan(
            input, output, init, storage, scan_op
        );

        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(
            input, output, init, reduction, storage, scan_op
        );
    }

    template<class PrefixCallback, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const auto warp_id = ::rocprim::warp_id(flat_tid);
        storage_type_& storage_ = storage.get();
        this->exclusive_scan_impl(
            flat_tid, input, output, storage, scan_op
        );

        T block_prefix = this->get_block_prefix(
            flat_tid, warp_id,
            storage_.warp_prefixes[warps_no_ - 1],
            prefix_callback_op, storage
        );
        output = scan_op(block_prefix, output);
        if(flat_tid == 0) output = block_prefix;
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {

        T thread_input = input[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }


        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input,
            init,
            storage,
            scan_op
        );


        T prev = input[0];
        T exclusive = init;
        if(flat_tid != 0)
        {
            exclusive = thread_input;
        }
        output[0] = exclusive;

        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            exclusive = scan_op(exclusive, prev);
            prev = input[i];
            output[i] = exclusive;
        }
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(input, output, init, storage, scan_op);
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        this->exclusive_scan(input, output, init, storage, scan_op);

        reduction = storage_.warp_prefixes[warps_no_ - 1];
    }

    template<unsigned int ItemsPerThread, class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        BinaryFunction scan_op)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->exclusive_scan(input, output, init, reduction, storage, scan_op);
    }

    template<
        class PrefixCallback,
        unsigned int ItemsPerThread,
        class BinaryFunction
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();

        T thread_input = input[0];
        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            thread_input = scan_op(thread_input, input[i]);
        }


        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        this->exclusive_scan_impl(
            flat_tid,
            thread_input, thread_input,
            storage,
            scan_op
        );


        T block_prefix = this->get_block_prefix(
            flat_tid, ::rocprim::warp_id(flat_tid),
            storage_.warp_prefixes[warps_no_ - 1],
            prefix_callback_op, storage
        );


        T prev = input[0];
        T exclusive = block_prefix;
        if(flat_tid != 0)
        {
            exclusive = scan_op(block_prefix, thread_input);
        }
        output[0] = exclusive;

        ROCPRIM_UNROLL
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            exclusive = scan_op(exclusive, prev);
            prev = input[i];
            output[i] = exclusive;
        }
    }

private:
    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<(BlockSize_ > ::rocprim::device_warp_size())>::type
    {
        storage_type_& storage_ = storage.get();

        warp_scan_input_type().inclusive_scan(

            input, output, scan_op
        );


        const auto warp_id = ::rocprim::warp_id(flat_tid);
        this->calculate_warp_prefixes(flat_tid, warp_id, output, storage, scan_op);


        if(warp_id != 0)
        {
            auto warp_prefix = storage_.warp_prefixes[warp_id - 1];
            output = scan_op(warp_prefix, output);
        }
    }


    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan_impl(unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<!(BlockSize_ > ::rocprim::device_warp_size())>::type
    {
        (void) storage;
        (void) flat_tid;
        storage_type_& storage_ = storage.get();

        warp_scan_input_type().inclusive_scan(

            input, output, scan_op
        );

        if(flat_tid == BlockSize_ - 1)
        {
            storage_.warp_prefixes[0] = output;
        }
        ::rocprim::syncthreads();
    }


    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             T init,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<(BlockSize_ > ::rocprim::device_warp_size())>::type
    {
        storage_type_& storage_ = storage.get();

        warp_scan_input_type().inclusive_scan(

            input, output, scan_op
        );


        const auto warp_id = ::rocprim::warp_id(flat_tid);
        this->calculate_warp_prefixes(flat_tid, warp_id, output, storage, scan_op);



        auto warp_prefix = init;
        if(warp_id != 0)
        {
            warp_prefix = scan_op(init, storage_.warp_prefixes[warp_id-1]);
        }


        output = scan_op(warp_prefix, output);
        output = warp_shuffle_up(output, 1, warp_size_);
        if(::rocprim::lane_id() == 0)
        {
            output = warp_prefix;
        }
    }



    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             T init,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<!(BlockSize_ > ::rocprim::device_warp_size())>::type
    {
        (void) flat_tid;
        (void) storage;
        (void) init;
        storage_type_& storage_ = storage.get();

        warp_scan_input_type().inclusive_scan(

            input, output, scan_op
        );

        if(flat_tid == BlockSize_ - 1)
        {
            storage_.warp_prefixes[0] = output;
        }
        ::rocprim::syncthreads();


        output = scan_op(init, output);
        output = warp_shuffle_up(output, 1, warp_size_);
        if(::rocprim::lane_id() == 0)
        {
            output = init;
        }
    }


    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<(BlockSize_ > ::rocprim::device_warp_size())>::type
    {
        storage_type_& storage_ = storage.get();

        warp_scan_input_type().inclusive_scan(

            input, output, scan_op
        );


        const auto warp_id = ::rocprim::warp_id(flat_tid);
        this->calculate_warp_prefixes(flat_tid, warp_id, output, storage, scan_op);


        T warp_prefix;
        if(warp_id != 0)
        {
            warp_prefix = storage_.warp_prefixes[warp_id - 1];
            output = scan_op(warp_prefix, output);
        }
        output = warp_shuffle_up(output, 1, warp_size_);
        if(::rocprim::lane_id() == 0)
        {
            output = warp_prefix;
        }
    }



    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan_impl(const unsigned int flat_tid,
                             T input,
                             T& output,
                             storage_type& storage,
                             BinaryFunction scan_op)
        -> typename std::enable_if<!(BlockSize_ > ::rocprim::device_warp_size())>::type
    {
        (void) flat_tid;
        (void) storage;
        storage_type_& storage_ = storage.get();

        warp_scan_input_type().inclusive_scan(

            input, output, scan_op
        );

        if(flat_tid == BlockSize_ - 1)
        {
            storage_.warp_prefixes[0] = output;
        }
        ::rocprim::syncthreads();
        output = warp_shuffle_up(output, 1, warp_size_);
    }


    template<class BinaryFunction, unsigned int BlockSize_ = BlockSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void calculate_warp_prefixes(const unsigned int flat_tid,
                                 const unsigned int warp_id,
                                 T inclusive_input,
                                 storage_type& storage,
                                 BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();


        if(flat_tid == ::rocprim::min((warp_id+1) * warp_size_, BlockSize_) - 1)
        {
            storage_.warp_prefixes[warp_id] = inclusive_input;
        }
        ::rocprim::syncthreads();


        if(flat_tid < warps_no_)
        {
            auto warp_prefix = storage_.warp_prefixes[flat_tid];
            warp_scan_prefix_type().inclusive_scan(

                warp_prefix, warp_prefix, scan_op
            );
            storage_.warp_prefixes[flat_tid] = warp_prefix;
        }
        ::rocprim::syncthreads();
    }


    template<class PrefixCallback>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_block_prefix(const unsigned int flat_tid,
                       const unsigned int warp_id,
                       const T reduction,
                       PrefixCallback& prefix_callback_op,
                       storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        if(warp_id == 0)
        {
            T block_prefix = prefix_callback_op(reduction);
            if(flat_tid == 0)
            {

                storage_.warp_prefixes[warps_no_ - 1] = block_prefix;
            }
        }
        ::rocprim::syncthreads();
        return storage_.warp_prefixes[warps_no_ - 1];
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
