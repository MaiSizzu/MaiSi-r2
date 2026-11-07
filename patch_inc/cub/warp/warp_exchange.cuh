

#ifndef HIPCUB_ROCPRIM_WARP_WARP_EXCHANGE_HPP_
#define HIPCUB_ROCPRIM_WARP_WARP_EXCHANGE_HPP_

#include "../config.hpp"
#include "../util_type.cuh"

#include <cub/rocprim/warp/warp_exchange.hpp>

BEGIN_HIPCUB_NAMESPACE

template <
    typename InputT,
    int ITEMS_PER_THREAD,
    int LOGICAL_WARP_THREADS = HIPCUB_DEVICE_WARP_THREADS,
    int ARCH = HIPCUB_ARCH
>
class WarpExchange
{
    using base_type = typename rocprim::warp_exchange<InputT, ITEMS_PER_THREAD, LOGICAL_WARP_THREADS>;

public:
    using TempStorage = typename base_type::storage_type;

private:
    TempStorage &temp_storage;

public:
    WarpExchange() = delete;

    explicit HIPCUB_DEVICE __forceinline__
    WarpExchange(TempStorage &temp_storage) :
        temp_storage(temp_storage)
    {
    }

    template <typename OutputT>
    HIPCUB_DEVICE __forceinline__
    void BlockedToStriped(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type rocprim_warp_exchange;
        rocprim_warp_exchange.blocked_to_striped(input_items, output_items, temp_storage);
    }

    template <typename OutputT>
    HIPCUB_DEVICE __forceinline__
    void StripedToBlocked(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type rocprim_warp_exchange;
        rocprim_warp_exchange.striped_to_blocked(input_items, output_items, temp_storage);
    }

    template <typename OffsetT>
    HIPCUB_DEVICE __forceinline__
    void ScatterToStriped(
        InputT (&items)[ITEMS_PER_THREAD],
        OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(items, items, ranks);
    }

    template <typename OutputT,
              typename OffsetT>
    HIPCUB_DEVICE __forceinline__
    void ScatterToStriped(
        const InputT (&input_items)[ITEMS_PER_THREAD],
        OutputT (&output_items)[ITEMS_PER_THREAD],
        OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type rocprim_warp_exchange;
        rocprim_warp_exchange.scatter_to_striped(input_items, output_items, ranks, temp_storage);
    }
};

END_HIPCUB_NAMESPACE

#endif
