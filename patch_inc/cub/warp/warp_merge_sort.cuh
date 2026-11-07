

#ifndef HIPCUB_ROCPRIM_WARP_WARP_MERGE_SORT_
#define HIPCUB_ROCPRIM_WARP_WARP_MERGE_SORT_

#include "../config.hpp"

#include "../block/block_merge_sort.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"

#include <cub/rocprim/functional.hpp>
#include <cub/rocprim/intrinsics/thread.hpp>

BEGIN_HIPCUB_NAMESPACE


template <
  typename    KeyT,
  int         ITEMS_PER_THREAD,
  int         LOGICAL_WARP_THREADS    = HIPCUB_DEVICE_WARP_THREADS,
  typename    ValueT                  = NullType,
  int         PTX_ARCH                = HIPCUB_ARCH>
class WarpMergeSort
    : public BlockMergeSortStrategy<
        KeyT,
        ValueT,
        LOGICAL_WARP_THREADS,
        ITEMS_PER_THREAD,
        WarpMergeSort<KeyT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS, ValueT, PTX_ARCH>>
{
private:
  constexpr static bool IS_ARCH_WARP = LOGICAL_WARP_THREADS == HIPCUB_DEVICE_WARP_THREADS;
  constexpr static bool KEYS_ONLY = ::rocprim::Equals<ValueT, NullType>::VALUE;
  constexpr static int TILE_SIZE = ITEMS_PER_THREAD * LOGICAL_WARP_THREADS;

  using BlockMergeSortStrategyT = BlockMergeSortStrategy<KeyT,
                                                         ValueT,
                                                         LOGICAL_WARP_THREADS,
                                                         ITEMS_PER_THREAD,
                                                         WarpMergeSort>;

  const unsigned int warp_id;
  const uint64_t member_mask;

public:
  WarpMergeSort() = delete;

  HIPCUB_DEVICE __forceinline__
  WarpMergeSort(typename BlockMergeSortStrategyT::TempStorage &temp_storage)
      : BlockMergeSortStrategyT(temp_storage,
                                IS_ARCH_WARP
                                  ? LaneId()
                                  : (LaneId() % LOGICAL_WARP_THREADS))
      , warp_id(IS_ARCH_WARP ? 0 : (LaneId() / LOGICAL_WARP_THREADS))
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(warp_id))
  {
  }

  HIPCUB_DEVICE __forceinline__ uint64_t get_member_mask() const
  {
    return member_mask;
  }

private:
  HIPCUB_DEVICE __forceinline__ void SyncImplementation() const
  {
    WARP_SYNC(member_mask);
  }

  friend BlockMergeSortStrategyT;
};


END_HIPCUB_NAMESPACE

#endif
