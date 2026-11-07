

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_LOAD_FUNC_HPP_

#include "../config.hpp"

#include <cub/rocprim/block/block_load_func.hpp>

BEGIN_HIPCUB_NAMESPACE

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectBlocked(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_blocked(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectBlocked(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items)
{
    ::rocprim::block_load_direct_blocked(
        linear_id, block_iter, items, valid_items
    );
}

template<
    typename T,
    typename Default,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectBlocked(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items,
                       Default oob_default)
{
    ::rocprim::block_load_direct_blocked(
        linear_id, block_iter, items, valid_items, oob_default
    );
}

template <
    typename T,
    int ITEMS_PER_THREAD
>
HIPCUB_DEVICE inline
void LoadDirectBlockedVectorized(int linear_id,
                                 T* block_iter,
                                 T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_blocked_vectorized(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectStriped(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectStriped(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items)
{
    ::rocprim::block_load_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items, valid_items
    );
}

template<
    int BLOCK_THREADS,
    typename T,
    typename Default,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectStriped(int linear_id,
                       InputIteratorT block_iter,
                       T (&items)[ITEMS_PER_THREAD],
                       int valid_items,
                       Default oob_default)
{
    ::rocprim::block_load_direct_striped<BLOCK_THREADS>(
        linear_id, block_iter, items, valid_items, oob_default
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectWarpStriped(int linear_id,
                           InputIteratorT block_iter,
                           T (&items)[ITEMS_PER_THREAD])
{
    ::rocprim::block_load_direct_warp_striped(
        linear_id, block_iter, items
    );
}

template<
    typename T,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectWarpStriped(int linear_id,
                           InputIteratorT block_iter,
                           T (&items)[ITEMS_PER_THREAD],
                           int valid_items)
{
    ::rocprim::block_load_direct_warp_striped(
        linear_id, block_iter, items, valid_items
    );
}

template<
    typename T,
    typename Default,
    int ITEMS_PER_THREAD,
    typename InputIteratorT
>
HIPCUB_DEVICE inline
void LoadDirectWarpStriped(int linear_id,
                           InputIteratorT block_iter,
                           T (&items)[ITEMS_PER_THREAD],
                           int valid_items,
                           Default oob_default)
{
    ::rocprim::block_load_direct_warp_striped(
        linear_id, block_iter, items, valid_items, oob_default
    );
}

END_HIPCUB_NAMESPACE

#endif
