

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_FUNC_HPP_

#include "../config.hpp"

#include <cub/rocprim/block/block_store_func.hpp>

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectBlocked(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_blocked(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectBlocked(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD],
                        int valid_items)
{
    ::rocprim::block_store_direct_blocked(
        linear_id, block_iter, items, valid_items
    );
}

template <
    typename T,
    int ITEMS_PER_THREAD
>
HIPCUB_DEVICE inline
void StoreDirectBlockedVectorized(int linear_id,
                                  T* block_iter,
                                  T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_blocked_vectorized(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectStriped(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectStriped(int linear_id,
                        OutputIteratorT block_iter,
                        T (&items)[ITEMS_PER_THREAD],
                        int valid_items)
{
    ::rocprim::block_store_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items, valid_items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectWarpStriped(int linear_id,
                            OutputIteratorT block_iter,
                            T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_store_direct_warp_striped(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename OutputIteratorT
>
HIPCUB_DEVICE inline
void StoreDirectWarpStriped(int linear_id,
                            OutputIteratorT block_iter,
                            T (&items)[ITEMS_PER_THREAD],
                            int valid_items)
{
    ::rocprim::block_store_direct_warp_striped(
        linear_id, block_iter, items, valid_items
    );
}

END_HIPCUB_NAMESPACE

#endif
