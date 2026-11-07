

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_STORE_HPP_

#include <type_traits>

#include "../config.hpp"

#include "block_store_func.hpp"

#include <cub/rocprim/block/block_store.hpp>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
    inline constexpr
    typename std::underlying_type<::rocprim::block_store_method>::type
    to_BlockStoreAlgorithm_enum(::rocprim::block_store_method v)
    {
        using utype = std::underlying_type<::rocprim::block_store_method>::type;
        return static_cast<utype>(v);
    }
}

enum BlockStoreAlgorithm
{
    BLOCK_STORE_DIRECT
        = detail::to_BlockStoreAlgorithm_enum(::rocprim::block_store_method::block_store_direct),
    BLOCK_STORE_STRIPED
        = detail::to_BlockStoreAlgorithm_enum(::rocprim::block_store_method::block_store_striped),
    BLOCK_STORE_VECTORIZE
        = detail::to_BlockStoreAlgorithm_enum(::rocprim::block_store_method::block_store_vectorize),
    BLOCK_STORE_TRANSPOSE
        = detail::to_BlockStoreAlgorithm_enum(::rocprim::block_store_method::block_store_transpose),
    BLOCK_STORE_WARP_TRANSPOSE
        = detail::to_BlockStoreAlgorithm_enum(::rocprim::block_store_method::block_store_warp_transpose),
    BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED
        = detail::to_BlockStoreAlgorithm_enum(::rocprim::block_store_method::block_store_warp_transpose)
};

template<
    typename T,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    BlockStoreAlgorithm ALGORITHM = BLOCK_STORE_DIRECT,
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockStore
    : private ::rocprim::block_store<
        T,
        BLOCK_DIM_X,
        ITEMS_PER_THREAD,
        static_cast<::rocprim::block_store_method>(ALGORITHM),
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_store<
            T,
            BLOCK_DIM_X,
            ITEMS_PER_THREAD,
            static_cast<::rocprim::block_store_method>(ALGORITHM),
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockStore() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockStore(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<class OutputIteratorT>
    HIPCUB_DEVICE inline
    void Store(OutputIteratorT block_iter,
               T (&items)[ITEMS_PER_THREAD])
    {
        base_type::store(block_iter, items, temp_storage_);
    }

    template<class OutputIteratorT>
    HIPCUB_DEVICE inline
    void Store(OutputIteratorT block_iter,
               T (&items)[ITEMS_PER_THREAD],
               int valid_items)
    {
        base_type::store(block_iter, items, valid_items, temp_storage_);
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
