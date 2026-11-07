

#ifndef HIPCUB_ROCPRIM_WARP_WARP_LOAD_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_LOAD_HPP_

#include "../config.hpp"

#include "../util_type.cuh"
#include "../iterator/cache_modified_input_iterator.cuh"
#include "./warp_exchange.cuh"

#include <cub/rocprim/block/block_load_func.hpp>

BEGIN_HIPCUB_NAMESPACE

enum WarpLoadAlgorithm
{
    WARP_LOAD_DIRECT,
    WARP_LOAD_STRIPED,
    WARP_LOAD_VECTORIZE,
    WARP_LOAD_TRANSPOSE
};

template<
    class InputT,
    int ITEMS_PER_THREAD,
    WarpLoadAlgorithm ALGORITHM = WARP_LOAD_DIRECT,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH
>
class WarpLoad
{
private:
    constexpr static bool IS_ARCH_WARP 
        = static_cast<unsigned>(LOGICAL_WARP_THREADS) == HIPCUB_DEVICE_WARP_THREADS;

    template <WarpLoadAlgorithm _POLICY>
    struct LoadInternal;

    template <>
    struct LoadInternal<WARP_LOAD_DIRECT>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__
        LoadInternal(
            TempStorage & ,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
        }
    };

    template <>
    struct LoadInternal<WARP_LOAD_STRIPED>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__
        LoadInternal(
            TempStorage & ,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
        }
    };

    template <>
    struct LoadInternal<WARP_LOAD_VECTORIZE>
    {
        using TempStorage = NullType;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ LoadInternal(
            TempStorage & ,
            int linear_tid)
            : linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputT *block_ptr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_ptr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            const InputT *block_ptr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_ptr,
                items
            );
        }

        template<
            CacheLoadModifier MODIFIER,
            typename ValueType,
            typename OffsetT
        >
        HIPCUB_DEVICE __forceinline__ void Load(
            CacheModifiedInputIterator<MODIFIER, ValueType, OffsetT> block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename _InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            _InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_blocked_vectorized(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {


            ::rocprim::block_load_direct_blocked(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
        }
    };

    template <>
    struct LoadInternal<WARP_LOAD_TRANSPOSE>
    {
        using WarpExchangeT = WarpExchange<
            InputT,
            ITEMS_PER_THREAD,
            LOGICAL_WARP_THREADS,
            ARCH
        >;
        using TempStorage = typename WarpExchangeT::TempStorage;
        TempStorage& temp_storage;
        int linear_tid;

        HIPCUB_DEVICE __forceinline__ LoadInternal(
            TempStorage &temp_storage,
            int linear_tid) :
            temp_storage(temp_storage),
            linear_tid(linear_tid)
        {
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD])
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items
            );
            WarpExchangeT(temp_storage).StripedToBlocked(items, items);
        }

        template <typename InputIteratorT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items)
            );
            WarpExchangeT(temp_storage).StripedToBlocked(items, items);
        }

        template <typename InputIteratorT, typename DefaultT>
        HIPCUB_DEVICE __forceinline__ void Load(
            InputIteratorT block_itr,
            InputT (&items)[ITEMS_PER_THREAD],
            int valid_items,
            DefaultT oob_default)
        {
            ::rocprim::block_load_direct_warp_striped<LOGICAL_WARP_THREADS>(
                static_cast<unsigned>(linear_tid),
                block_itr,
                items,
                static_cast<unsigned>(valid_items),
                oob_default
            );
            WarpExchangeT(temp_storage).StripedToBlocked(items, items);
        }
    };

    using InternalLoad = LoadInternal<ALGORITHM>;

    using _TempStorage = typename InternalLoad::TempStorage;

    HIPCUB_DEVICE __forceinline__ _TempStorage &PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

    _TempStorage &temp_storage;
    int linear_tid;

public:
    struct TempStorage : Uninitialized<_TempStorage>
    {
    };

    HIPCUB_DEVICE __forceinline__
    WarpLoad() :
        temp_storage(PrivateStorage()),
        linear_tid(IS_ARCH_WARP ? ::rocprim::lane_id() : (::rocprim::lane_id() % LOGICAL_WARP_THREADS))
    {
    }

    HIPCUB_DEVICE __forceinline__
    WarpLoad(TempStorage &temp_storage) :
        temp_storage(temp_storage.Alias()),
        linear_tid(IS_ARCH_WARP ? ::rocprim::lane_id() : (::rocprim::lane_id() % LOGICAL_WARP_THREADS))
    {
    }

    template <typename InputIteratorT>
    HIPCUB_DEVICE __forceinline__ void Load(
        InputIteratorT block_itr,
        InputT (&items)[ITEMS_PER_THREAD])
    {
        InternalLoad(temp_storage, linear_tid)
            .Load(block_itr, items);
    }

    template <typename InputIteratorT>
    HIPCUB_DEVICE __forceinline__ void Load(
        InputIteratorT block_itr,
        InputT (&items)[ITEMS_PER_THREAD],
        int valid_items)
    {
        InternalLoad(temp_storage, linear_tid)
            .Load(block_itr, items, valid_items);
    }

    template <typename InputIteratorT,
              typename DefaultT>
    HIPCUB_DEVICE __forceinline__ void Load(
        InputIteratorT block_itr,
        InputT (&items)[ITEMS_PER_THREAD],
        int valid_items,
        DefaultT oob_default)
    {
        InternalLoad(temp_storage, linear_tid)
            .Load(block_itr, items, valid_items, oob_default);
    }
};

END_HIPCUB_NAMESPACE

#endif
