

#ifndef HIPCUB_ROCPRIM_THREAD_SORT_HPP_
#define HIPCUB_ROCPRIM_THREAD_SORT_HPP_

#include "../config.hpp"

#include "../util_ptx.cuh"
#include "../util_type.cuh"

#include <cub/rocprim/functional.hpp>

BEGIN_HIPCUB_NAMESPACE


template <typename T>
HIPCUB_DEVICE __forceinline__ void Swap(T &lhs, T &rhs)
{
  T temp = lhs;
  lhs    = rhs;
  rhs    = temp;
}



template <typename KeyT,
          typename ValueT,
          typename CompareOp,
          int ITEMS_PER_THREAD>
HIPCUB_DEVICE __forceinline__ void
StableOddEvenSort(KeyT (&keys)[ITEMS_PER_THREAD],
                  ValueT (&items)[ITEMS_PER_THREAD],
                  CompareOp compare_op)
{
  constexpr bool KEYS_ONLY = ::rocprim::Equals<ValueT, NullType>::VALUE;

  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
  #pragma unroll
    for (int j = 1 & i; j < ITEMS_PER_THREAD - 1; j += 2)
    {
      if (compare_op(keys[j + 1], keys[j]))
      {
        Swap(keys[j], keys[j + 1]);
        if (!KEYS_ONLY)
        {
          Swap(items[j], items[j + 1]);
        }
      }
    }
  }
}


END_HIPCUB_NAMESPACE

#endif
