

#ifndef ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
#define ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_reduce_warp_reduce.hpp"
#include "detail/block_reduce_raking_reduce.hpp"




BEGIN_ROCPRIM_NAMESPACE










































template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1>
class block_shuffle
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;


    struct storage_type_
    {
        T prev[BlockSize];
        T next[BlockSize];
    };

public:









    #ifndef DOXYGEN_SHOULD_SKIP_THIS
        using storage_type = detail::raw_storage<storage_type_>;
    #else
        using storage_type = storage_type_;
    #endif



























    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void offset(T input,
                T& output,
                int distance = 1)
    {
        offset(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, distance
        );
    }

    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void offset(const size_t& flat_id,
                T input,
                T& output,
                int distance)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        offset(flat_id, input, output, distance, storage);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void offset(const size_t& flat_id,
                T input,
                T& output,
                int distance,
                storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.prev[flat_id] = input;

        ::rocprim::syncthreads();

        const int offset_tid = static_cast<int>(flat_id) + distance;
        if ((offset_tid >= 0) && (offset_tid < (int)BlockSize))
        {
            output = storage_.prev[static_cast<size_t>(offset_tid)];
        }
    }



























    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void rotate(T input,
                T& output,
                unsigned int distance = 1)
    {
        rotate(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, output, distance
        );
    }

    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void rotate(const size_t& flat_id,
                T input,
                T& output,
                unsigned int distance)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        rotate(flat_id, input, output, distance, storage);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void rotate(const size_t& flat_id,
                T input,
                T& output,
                unsigned int distance,
                storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.prev[flat_id] = input;

        ::rocprim::syncthreads();

        unsigned int offset = threadIdx.x + distance;
        if (offset >= BlockSize)
            offset -= BlockSize;

        output = storage_.prev[offset];
    }

























    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void up(T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread])
    {
        this->up(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, prev
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->up(flat_id, input, prev, storage);
    }


    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.prev[flat_id] = input[ItemsPerThread -1];

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for (unsigned int i = ItemsPerThread - 1; i > 0; --i)
        {
            prev[i] = input[i - 1];
        }

        if (flat_id > 0)
        {
            prev[0] = storage_.prev[flat_id - 1];
        }
    }











    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void up(T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            T &block_suffix)
    {
        this->up(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, prev, block_suffix
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            T &block_suffix)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->up(flat_id, input, prev, block_suffix, storage);
    }

    template <int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void up(const size_t& flat_id,
            T (&input)[ItemsPerThread],
            T (&prev)[ItemsPerThread],
            T &block_suffix,
            storage_type& storage)
    {
        up(flat_id, input, prev, storage);


        block_suffix = storage->prev[BlockSize - 1];
    }
























    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void down(T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread])
    {
        this->down(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, next
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->down(flat_id, input, next, storage);
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        storage_.next[flat_id] = input[0];

        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < (ItemsPerThread - 1); ++i)
        {
          next[i] = input[i + 1];
        }

        if (flat_id <(BlockSize -1))
        {
          next[ItemsPerThread -1] = storage_.next[flat_id + 1];
        }
    }








    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void down(T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              T &block_prefix)
    {
        this->down(
            ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>(),
            input, next, block_prefix
        );
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              T &block_prefix)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->down(flat_id, input, next, block_prefix, storage);
    }

    template <unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void down(const size_t& flat_id,
              T (&input)[ItemsPerThread],
              T (&next)[ItemsPerThread],
              T &block_prefix,
              storage_type& storage)
    {
        this->down(flat_id, input, next, storage);


        block_prefix = storage->next[0];
    }
};


END_ROCPRIM_NAMESPACE




#endif
