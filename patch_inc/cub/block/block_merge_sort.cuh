

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_MERGE_SORT_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_MERGE_SORT_HPP_

#include "../thread/thread_sort.cuh"
#include "../util_math.cuh"
#include "../util_type.cuh"

#include <cub/rocprim/detail/various.hpp>
#include <cub/rocprim/functional.hpp>

BEGIN_HIPCUB_NAMESPACE







template <typename KeyT,
        typename KeyIteratorT,
        typename OffsetT,
        typename BinaryPred>
HIPCUB_DEVICE __forceinline__ OffsetT MergePath(KeyIteratorT keys1,
                                            KeyIteratorT keys2,
                                            OffsetT keys1_count,
                                            OffsetT keys2_count,
                                            OffsetT diag,
                                            BinaryPred binary_pred)
{
   OffsetT keys1_begin = diag < keys2_count ? 0 : diag - keys2_count;
   OffsetT keys1_end   = (::rocprim::min)(diag, keys1_count);

   while (keys1_begin < keys1_end)
   {
       OffsetT mid = cub::MidPoint<OffsetT>(keys1_begin, keys1_end);
       KeyT key1   = keys1[mid];
       KeyT key2   = keys2[diag - 1 - mid];
       bool pred   = binary_pred(key2, key1);

       if (pred)
       {
           keys1_end = mid;
       }
       else
       {
           keys1_begin = mid + 1;
       }
   }
   return keys1_begin;
}

template <typename KeyT, typename CompareOp, int ITEMS_PER_THREAD>
HIPCUB_DEVICE __forceinline__ void SerialMerge(KeyT *keys_shared,
                                           int keys1_beg,
                                           int keys2_beg,
                                           int keys1_count,
                                           int keys2_count,
                                           KeyT (&output)[ITEMS_PER_THREAD],
                                           int (&indices)[ITEMS_PER_THREAD],
                                           CompareOp compare_op)
{
   int keys1_end = keys1_beg + keys1_count;
   int keys2_end = keys2_beg + keys2_count;

   KeyT key1 = keys_shared[keys1_beg];
   KeyT key2 = keys_shared[keys2_beg];

#pragma unroll
   for (int item = 0; item < ITEMS_PER_THREAD; ++item)
   {
       bool p = (keys2_beg < keys2_end) &&
                ((keys1_beg >= keys1_end)
                 || compare_op(key2, key1));

       output[item]  = p ? key2 : key1;
       indices[item] = p ? keys2_beg++ : keys1_beg++;

       if (p)
       {
           key2 = keys_shared[keys2_beg];
       }
       else
       {
           key1 = keys_shared[keys1_beg];
       }
   }
}


template <typename KeyT,
          typename ValueT,
          int NUM_THREADS,
          int ITEMS_PER_THREAD,
          typename SynchronizationPolicy>
class BlockMergeSortStrategy
{
  static_assert(PowerOfTwo<NUM_THREADS>::VALUE,
                "NUM_THREADS must be a power of two");

private:

  static constexpr int ITEMS_PER_TILE = ITEMS_PER_THREAD * NUM_THREADS;


  static constexpr bool KEYS_ONLY = ::rocprim::Equals<ValueT, NullType>::VALUE;


  union _TempStorage
  {
    KeyT keys_shared[ITEMS_PER_TILE + 1];
    ValueT items_shared[ITEMS_PER_TILE + 1];
  };


  _TempStorage &temp_storage;


  HIPCUB_DEVICE __forceinline__ _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  const unsigned int linear_tid;

public:

  struct TempStorage : Uninitialized<_TempStorage> {};

  BlockMergeSortStrategy() = delete;
  explicit HIPCUB_DEVICE __forceinline__
  BlockMergeSortStrategy(unsigned int linear_tid)
      : temp_storage(PrivateStorage())
      , linear_tid(linear_tid)
  {}

  HIPCUB_DEVICE __forceinline__ BlockMergeSortStrategy(TempStorage &temp_storage,
                                                    unsigned int linear_tid)
      : temp_storage(temp_storage.Alias())
      , linear_tid(linear_tid)
  {}

  HIPCUB_DEVICE __forceinline__ unsigned int get_linear_tid() const
  {
    return linear_tid;
  }

  
  template <typename CompareOp>
  HIPCUB_DEVICE __forceinline__ void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
                                       CompareOp compare_op)
  {
    ValueT items[ITEMS_PER_THREAD];
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }

  
  template <typename CompareOp>
  HIPCUB_DEVICE __forceinline__ void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
                                       CompareOp compare_op,
                                       int valid_items,
                                       KeyT oob_default)
  {
    ValueT items[ITEMS_PER_THREAD];
    Sort<CompareOp, true>(keys, items, compare_op, valid_items, oob_default);
  }

  
  template <typename CompareOp>
  HIPCUB_DEVICE __forceinline__ void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
                                       ValueT (&items)[ITEMS_PER_THREAD],
                                       CompareOp compare_op)
  {
    Sort<CompareOp, false>(keys, items, compare_op, ITEMS_PER_TILE, keys[0]);
  }

  
  template <typename CompareOp,
            bool IS_LAST_TILE = true>
  HIPCUB_DEVICE __forceinline__ void Sort(KeyT (&keys)[ITEMS_PER_THREAD],
                                       ValueT (&items)[ITEMS_PER_THREAD],
                                       CompareOp compare_op,
                                       int valid_items,
                                       KeyT oob_default)
  {
    if (IS_LAST_TILE)
    {



      KeyT max_key = oob_default;

      #pragma unroll
      for (int item = 1; item < ITEMS_PER_THREAD; ++item)
      {
        if (ITEMS_PER_THREAD * static_cast<int>(linear_tid) + item < valid_items)
        {
          max_key = compare_op(max_key, keys[item]) ? keys[item] : max_key;
        }
        else
        {
          keys[item] = max_key;
        }
      }
    }



    if (!IS_LAST_TILE || ITEMS_PER_THREAD * static_cast<int>(linear_tid) < valid_items)
    {
      StableOddEvenSort(keys, items, compare_op);
    }




    #pragma unroll
    for (int target_merged_threads_number = 2;
         target_merged_threads_number <= NUM_THREADS;
         target_merged_threads_number *= 2)
    {
      int merged_threads_number = target_merged_threads_number / 2;
      int mask = target_merged_threads_number - 1;

      Sync();



      #pragma unroll
      for (int item = 0; item < ITEMS_PER_THREAD; ++item)
      {
        int idx                       = ITEMS_PER_THREAD * linear_tid + item;
        temp_storage.keys_shared[idx] = keys[item];
      }

      Sync();

      int indices[ITEMS_PER_THREAD];

      int first_thread_idx_in_thread_group_being_merged = ~mask & linear_tid;
      int start = ITEMS_PER_THREAD * first_thread_idx_in_thread_group_being_merged;
      int size  = ITEMS_PER_THREAD * merged_threads_number;

      int thread_idx_in_thread_group_being_merged = mask & linear_tid;

      int diag =
        (::rocprim::min)(valid_items,
                   ITEMS_PER_THREAD * thread_idx_in_thread_group_being_merged);

      int keys1_beg = (::rocprim::min)(valid_items, start);
      int keys1_end = (::rocprim::min)(valid_items, keys1_beg + size);
      int keys2_beg = keys1_end;
      int keys2_end = (::rocprim::min)(valid_items, keys2_beg + size);

      int keys1_count = keys1_end - keys1_beg;
      int keys2_count = keys2_end - keys2_beg;

      int partition_diag = MergePath<KeyT>(&temp_storage.keys_shared[keys1_beg],
                                           &temp_storage.keys_shared[keys2_beg],
                                           keys1_count,
                                           keys2_count,
                                           diag,
                                           compare_op);

      int keys1_beg_loc   = keys1_beg + partition_diag;
      int keys1_end_loc   = keys1_end;
      int keys2_beg_loc   = keys2_beg + diag - partition_diag;
      int keys2_end_loc   = keys2_end;
      int keys1_count_loc = keys1_end_loc - keys1_beg_loc;
      int keys2_count_loc = keys2_end_loc - keys2_beg_loc;
      SerialMerge(&temp_storage.keys_shared[0],
                  keys1_beg_loc,
                  keys2_beg_loc,
                  keys1_count_loc,
                  keys2_count_loc,
                  keys,
                  indices,
                  compare_op);

      if (!KEYS_ONLY)
      {
        Sync();



        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
          int idx = ITEMS_PER_THREAD * linear_tid + item;
          temp_storage.items_shared[idx] = items[item];
        }

        Sync();



        #pragma unroll
        for (int item = 0; item < ITEMS_PER_THREAD; ++item)
        {
          items[item] = temp_storage.items_shared[indices[item]];
        }
      }
    }
  }

  
  template <typename CompareOp>
  HIPCUB_DEVICE __forceinline__ void StableSort(KeyT (&keys)[ITEMS_PER_THREAD],
                                             CompareOp compare_op)
  {
    Sort(keys, compare_op);
  }

  
  template <typename CompareOp>
  HIPCUB_DEVICE __forceinline__ void StableSort(KeyT (&keys)[ITEMS_PER_THREAD],
                                             ValueT (&items)[ITEMS_PER_THREAD],
                                             CompareOp compare_op)
  {
    Sort(keys, items, compare_op);
  }

  
  template <typename CompareOp>
  HIPCUB_DEVICE __forceinline__ void StableSort(KeyT (&keys)[ITEMS_PER_THREAD],
                                             CompareOp compare_op,
                                             int valid_items,
                                             KeyT oob_default)
  {
    Sort(keys, compare_op, valid_items, oob_default);
  }

  
  template <typename CompareOp,
            bool IS_LAST_TILE = true>
  HIPCUB_DEVICE __forceinline__ void StableSort(KeyT (&keys)[ITEMS_PER_THREAD],
                                             ValueT (&items)[ITEMS_PER_THREAD],
                                             CompareOp compare_op,
                                             int valid_items,
                                             KeyT oob_default)
  {
    Sort<CompareOp, IS_LAST_TILE>(keys,
                                  items,
                                  compare_op,
                                  valid_items,
                                  oob_default);
  }

private:
  HIPCUB_DEVICE __forceinline__ void Sync() const
  {
    static_cast<const SynchronizationPolicy*>(this)->SyncImplementation();
  }
};



template <typename KeyT,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD,
          typename ValueT = NullType,
          int BLOCK_DIM_Y = 1,
          int BLOCK_DIM_Z = 1>
class BlockMergeSort
    : public BlockMergeSortStrategy<KeyT,
                                    ValueT,
                                    BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
                                    ITEMS_PER_THREAD,
                                    BlockMergeSort<KeyT,
                                                   BLOCK_DIM_X,
                                                   ITEMS_PER_THREAD,
                                                   ValueT,
                                                   BLOCK_DIM_Y,
                                                   BLOCK_DIM_Z>>
{
private:

  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
  static constexpr int ITEMS_PER_TILE = ITEMS_PER_THREAD * BLOCK_THREADS;

  using BlockMergeSortStrategyT =
    BlockMergeSortStrategy<KeyT,
                           ValueT,
                           BLOCK_THREADS,
                           ITEMS_PER_THREAD,
                           BlockMergeSort>;

public:
  HIPCUB_DEVICE __forceinline__ BlockMergeSort()
      : BlockMergeSortStrategyT(
          RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

  HIPCUB_DEVICE __forceinline__ explicit BlockMergeSort(
    typename BlockMergeSortStrategyT::TempStorage &temp_storage)
      : BlockMergeSortStrategyT(
          temp_storage,
          RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
  {}

private:
  HIPCUB_DEVICE __forceinline__ void SyncImplementation() const
  {
    CTA_SYNC();
  }

  friend BlockMergeSortStrategyT;
};

END_HIPCUB_NAMESPACE

#endif
