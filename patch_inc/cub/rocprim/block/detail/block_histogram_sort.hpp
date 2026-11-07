



















#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_SORT_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_SORT_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

#include "../block_radix_sort.hpp"
#include "../block_discontinuity.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ,
    unsigned int ItemsPerThread,
    unsigned int Bins
>
class block_histogram_sort
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    static_assert(
        std::is_convertible<T, unsigned int>::value,
        "T must be convertible to unsigned int"
    );

private:
    using radix_sort = block_radix_sort<T, BlockSizeX, ItemsPerThread, empty_type, BlockSizeY, BlockSizeZ>;
    using discontinuity = block_discontinuity<T, BlockSizeX, BlockSizeY, BlockSizeZ>;

public:
    union storage_type_
    {
        typename radix_sort::storage_type sort;
        struct
        {
            typename discontinuity::storage_type flag;
            unsigned int start[Bins];
            unsigned int end[Bins];
        };
    };

    using storage_type = detail::raw_storage<storage_type_>;

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        this->composite(input, hist, storage);
    }

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {





        constexpr auto tile_size = BlockSize * ItemsPerThread;
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        unsigned int head_flags[ItemsPerThread];
        discontinuity_op flags_op(storage);
        storage_type_& storage_ = storage.get();

        radix_sort().sort(input, storage_.sort);
        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                storage_.start[offset_tid] = tile_size;
                storage_.end[offset_tid] = tile_size;
            }
        }
        ::rocprim::syncthreads();

        discontinuity().flag_heads(head_flags, input, flags_op, storage_.flag);
        ::rocprim::syncthreads();





        if(flat_tid == 0)
        {
            storage_.start[static_cast<unsigned int>(input[0])] = 0;
        }
        ::rocprim::syncthreads();

        ROCPRIM_UNROLL
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                Counter count = static_cast<Counter>(storage_.end[offset_tid] - storage_.start[offset_tid]);
                hist[offset_tid] += count;
            }
        }
    }

private:
    struct discontinuity_op
    {
        storage_type &storage;

        ROCPRIM_DEVICE ROCPRIM_INLINE
        discontinuity_op(storage_type &storage) : storage(storage)
        {
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE
        bool operator()(const T& a, const T& b, unsigned int b_index) const
        {
            storage_type_& storage_ = storage.get();
            if(static_cast<unsigned int>(a) != static_cast<unsigned int>(b))
            {
                storage_.start[static_cast<unsigned int>(b)] = b_index;
                storage_.end[static_cast<unsigned int>(a)] = b_index;
                return true;
            }
            else
            {
                return false;
            }
        }
    };
};

}

END_ROCPRIM_NAMESPACE

#endif
