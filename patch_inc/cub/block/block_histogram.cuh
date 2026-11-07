

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_HISTOGRAM_HPP_

#include <type_traits>

#include <cub/rocprim/block/block_histogram.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
    inline constexpr
    typename std::underlying_type<::rocprim::block_histogram_algorithm>::type
    to_BlockHistogramAlgorithm_enum(::rocprim::block_histogram_algorithm v)
    {
        using utype = std::underlying_type<::rocprim::block_histogram_algorithm>::type;
        return static_cast<utype>(v);
    }
}

enum BlockHistogramAlgorithm
{
    BLOCK_HISTO_ATOMIC
        = detail::to_BlockHistogramAlgorithm_enum(::rocprim::block_histogram_algorithm::using_atomic),
    BLOCK_HISTO_SORT
        = detail::to_BlockHistogramAlgorithm_enum(::rocprim::block_histogram_algorithm::using_sort)
};

template<
    typename T,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    int BINS,
    BlockHistogramAlgorithm ALGORITHM = BLOCK_HISTO_SORT,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockHistogram
    : private ::rocprim::block_histogram<
        T,
        BLOCK_DIM_X,
        ITEMS_PER_THREAD,
        BINS,
        static_cast<::rocprim::block_histogram_algorithm>(ALGORITHM),
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_histogram<
            T,
            BLOCK_DIM_X,
            ITEMS_PER_THREAD,
            BINS,
            static_cast<::rocprim::block_histogram_algorithm>(ALGORITHM),
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockHistogram() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockHistogram(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<class CounterT>
    HIPCUB_DEVICE inline
    void InitHistogram(CounterT histogram[BINS])
    {
        base_type::init_histogram(histogram);
    }

    template<class CounterT>
    HIPCUB_DEVICE inline
    void Composite(T (&items)[ITEMS_PER_THREAD],
                   CounterT histogram[BINS])
    {
        base_type::composite(items, histogram, temp_storage_);
    }

    template<class CounterT>
    HIPCUB_DEVICE inline
    void Histogram(T (&items)[ITEMS_PER_THREAD],
                   CounterT histogram[BINS])
    {
        base_type::init_histogram(histogram);
        CTA_SYNC();
        base_type::composite(items, histogram, temp_storage_);
    }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif
