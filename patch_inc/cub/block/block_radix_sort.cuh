

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include "../config.hpp"

#include "../util_type.cuh"

#include <cub/rocprim/functional.hpp>
#include <cub/rocprim/block/block_radix_sort.hpp>

#include "block_scan.cuh"

BEGIN_HIPCUB_NAMESPACE

template<
    typename KeyT,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    typename ValueT = NullType,
    int RADIX_BITS = 4, 
    bool MEMOIZE_OUTER_SCAN = true, 
    BlockScanAlgorithm INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS, 
    cudaSharedMemConfig SMEM_CONFIG = cudaSharedMemBankSizeFourByte, 
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int PTX_ARCH = HIPCUB_ARCH 
>
class BlockRadixSort
    : private ::rocprim::block_radix_sort<
        KeyT,
        BLOCK_DIM_X,
        ITEMS_PER_THREAD,
        ValueT,
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_radix_sort<
            KeyT,
            BLOCK_DIM_X,
            ITEMS_PER_THREAD,
            ValueT,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockRadixSort() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockRadixSort(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
              int begin_bit = 0,
              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
              ValueT (&values)[ITEMS_PER_THREAD],
              int begin_bit = 0,
              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort(keys, values, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                        int begin_bit = 0,
                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescending(KeyT (&keys)[ITEMS_PER_THREAD],
                        ValueT (&values)[ITEMS_PER_THREAD],
                        int begin_bit = 0,
                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc(keys, values, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                              int begin_bit = 0,
                              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_to_striped(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                              ValueT (&values)[ITEMS_PER_THREAD],
                              int begin_bit = 0,
                              int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_to_striped(keys, values, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                        int begin_bit = 0,
                                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc_to_striped(keys, temp_storage_, begin_bit, end_bit);
    }

    HIPCUB_DEVICE inline
    void SortDescendingBlockedToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                        ValueT (&values)[ITEMS_PER_THREAD],
                                        int begin_bit = 0,
                                        int end_bit = sizeof(KeyT) * 8)
    {
        base_type::sort_desc_to_striped(keys, values, temp_storage_, begin_bit, end_bit);
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
