

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_SCAN_HPP_

#include <type_traits>

#include "../config.hpp"

#include "../thread/thread_operators.cuh"

#include <cub/rocprim/block/block_scan.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
    inline constexpr
    typename std::underlying_type<::rocprim::block_scan_algorithm>::type
    to_BlockScanAlgorithm_enum(::rocprim::block_scan_algorithm v)
    {
        using utype = std::underlying_type<::rocprim::block_scan_algorithm>::type;
        return static_cast<utype>(v);
    }
}

enum BlockScanAlgorithm
{
    BLOCK_SCAN_RAKING
        = detail::to_BlockScanAlgorithm_enum(::rocprim::block_scan_algorithm::reduce_then_scan),
    BLOCK_SCAN_RAKING_MEMOIZE
        = detail::to_BlockScanAlgorithm_enum(::rocprim::block_scan_algorithm::reduce_then_scan),
    BLOCK_SCAN_WARP_SCANS
        = detail::to_BlockScanAlgorithm_enum(::rocprim::block_scan_algorithm::using_warp_scan)
};

template<
    typename T,
    int BLOCK_DIM_X,
    BlockScanAlgorithm ALGORITHM = BLOCK_SCAN_RAKING,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockScan
    : private ::rocprim::block_scan<
        T,
        BLOCK_DIM_X,
        static_cast<::rocprim::block_scan_algorithm>(ALGORITHM),
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_scan<
            T,
            BLOCK_DIM_X,
            static_cast<::rocprim::block_scan_algorithm>(ALGORITHM),
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockScan() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockScan(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& output)
    {
        base_type::inclusive_scan(input, output, temp_storage_);
    }

    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& output, T& block_aggregate)
    {
        base_type::inclusive_scan(input, output, block_aggregate, temp_storage_);
    }

    template<typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void InclusiveSum(T input, T& output, BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::inclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, cub::Sum()
        );
    }

    template<int ITEMS_PER_THREAD>
    HIPCUB_DEVICE inline
    void InclusiveSum(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD])
    {
        base_type::inclusive_scan(input, output, temp_storage_);
    }

    template<int ITEMS_PER_THREAD>
    HIPCUB_DEVICE inline
    void InclusiveSum(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                      T& block_aggregate)
    {
        base_type::inclusive_scan(input, output, block_aggregate, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void InclusiveSum(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                      BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::inclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, cub::Sum()
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& output, ScanOp scan_op)
    {
        base_type::inclusive_scan(input, output, temp_storage_, scan_op);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& output, ScanOp scan_op, T& block_aggregate)
    {
        base_type::inclusive_scan(input, output, block_aggregate, temp_storage_, scan_op);
    }

    template<typename ScanOp, typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T input, T& output, ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::inclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, scan_op
        );
    }

    template<int ITEMS_PER_THREAD, typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD], ScanOp scan_op)
    {
        base_type::inclusive_scan(input, output, temp_storage_, scan_op);
    }

    template<int ITEMS_PER_THREAD, typename ScanOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                       ScanOp scan_op, T& block_aggregate)
    {
        base_type::inclusive_scan(input, output, block_aggregate, temp_storage_, scan_op);
    }

    template<int ITEMS_PER_THREAD, typename ScanOp, typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void InclusiveScan(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                       ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::inclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, scan_op
        );
    }

    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& output)
    {
        base_type::exclusive_scan(input, output, T(0), temp_storage_);
    }

    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& output, T& block_aggregate)
    {
        base_type::exclusive_scan(input, output, T(0), block_aggregate, temp_storage_);
    }

    template<typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void ExclusiveSum(T input, T& output, BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::exclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, cub::Sum()
        );
    }

    template<int ITEMS_PER_THREAD>
    HIPCUB_DEVICE inline
    void ExclusiveSum(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD])
    {
        base_type::exclusive_scan(input, output, T(0), temp_storage_);
    }

    template<int ITEMS_PER_THREAD>
    HIPCUB_DEVICE inline
    void ExclusiveSum(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                      T& block_aggregate)
    {
        base_type::exclusive_scan(input, output, T(0), block_aggregate, temp_storage_);
    }

    template<int ITEMS_PER_THREAD, typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void ExclusiveSum(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                      BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::exclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, cub::Sum()
        );
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& output, T initial_value, ScanOp scan_op)
    {
        base_type::exclusive_scan(input, output, initial_value, temp_storage_, scan_op);
    }

    template<typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& output, T initial_value,
                       ScanOp scan_op, T& block_aggregate)
    {
        base_type::exclusive_scan(
            input, output, initial_value, block_aggregate, temp_storage_, scan_op
        );
    }

    template<typename ScanOp, typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T input, T& output, ScanOp scan_op,
                       BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::exclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, scan_op
        );
    }

    template<int ITEMS_PER_THREAD, typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                       T initial_value, ScanOp scan_op)
    {
        base_type::exclusive_scan(input, output, initial_value, temp_storage_, scan_op);
    }

    template<int ITEMS_PER_THREAD, typename ScanOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                       T initial_value, ScanOp scan_op, T& block_aggregate)
    {
        base_type::exclusive_scan(
            input, output, initial_value, block_aggregate, temp_storage_, scan_op
        );
    }

    template<int ITEMS_PER_THREAD, typename ScanOp, typename BlockPrefixCallbackOp>
    HIPCUB_DEVICE inline
    void ExclusiveScan(T(&input)[ITEMS_PER_THREAD], T(&output)[ITEMS_PER_THREAD],
                       ScanOp scan_op, BlockPrefixCallbackOp& block_prefix_callback_op)
    {
        base_type::exclusive_scan(
            input, output, temp_storage_, block_prefix_callback_op, scan_op
        );
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
