



















#ifndef ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
#define ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_histogram_atomic.hpp"
#include "detail/block_histogram_sort.hpp"

BEGIN_ROCPRIM_NAMESPACE





enum class block_histogram_algorithm
{





    using_atomic,







    using_sort,


    default_algorithm = using_atomic,
};

namespace detail
{



template<block_histogram_algorithm Algorithm>
struct select_block_histogram_impl;

template<>
struct select_block_histogram_impl<block_histogram_algorithm::using_atomic>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ, unsigned int ItemsPerThread, unsigned int Bins>
    using type = block_histogram_atomic<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>;
};

template<>
struct select_block_histogram_impl<block_histogram_algorithm::using_sort>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ, unsigned int ItemsPerThread, unsigned int Bins>
    using type = block_histogram_sort<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>;
};

}









































template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int Bins,
    block_histogram_algorithm Algorithm = block_histogram_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_histogram
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_histogram_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>
#endif
{
    using base_type = typename detail::select_block_histogram_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Bins>;
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
public:








    using storage_type = typename base_type::storage_type;






    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void init_histogram(Counter hist[Bins])
    {
        const auto flat_tid = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        ROCPRIM_UNROLL
        for(unsigned int offset = 0; offset < Bins; offset += BlockSize)
        {
            const unsigned int offset_tid = offset + flat_tid;
            if(offset_tid < Bins)
            {
                hist[offset_tid] = Counter();
            }
        }
    }
















































    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        base_type::composite(input, hist, storage);
    }












    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        base_type::composite(input, hist);
    }









































    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void histogram(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        init_histogram(hist);
        ::rocprim::syncthreads();
        composite(input, hist, storage);
    }












    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void histogram(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        init_histogram(hist);
        ::rocprim::syncthreads();
        composite(input, hist);
    }
};

END_ROCPRIM_NAMESPACE




#endif
