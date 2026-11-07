

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_

#include "../config.hpp"

#include <cub/rocprim/block/block_exchange.hpp>

BEGIN_HIPCUB_NAMESPACE

template<
    typename InputT,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    bool WARP_TIME_SLICING = false, 
    int BLOCK_DIM_Y = 1,
    int BLOCK_DIM_Z = 1,
    int ARCH = HIPCUB_ARCH 
>
class BlockExchange
    : private ::rocprim::block_exchange<
        InputT,
        BLOCK_DIM_X,
        ITEMS_PER_THREAD,
        BLOCK_DIM_Y,
        BLOCK_DIM_Z
      >
{
    static_assert(
        BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
        "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
    );

    using base_type =
        typename ::rocprim::block_exchange<
            InputT,
            BLOCK_DIM_X,
            ITEMS_PER_THREAD,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z
        >;


    typename base_type::storage_type& temp_storage_;

public:
    using TempStorage = typename base_type::storage_type;

    HIPCUB_DEVICE inline
    BlockExchange() : temp_storage_(private_storage())
    {
    }

    HIPCUB_DEVICE inline
    BlockExchange(TempStorage& temp_storage) : temp_storage_(temp_storage)
    {
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void StripedToBlocked(InputT  (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::striped_to_blocked(input_items, output_items, temp_storage_);
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void BlockedToStriped(InputT  (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::blocked_to_striped(input_items, output_items, temp_storage_);
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void WarpStripedToBlocked(InputT  (&input_items)[ITEMS_PER_THREAD],
                              OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::warp_striped_to_blocked(input_items, output_items, temp_storage_);
    }

    template<typename OutputT>
    HIPCUB_DEVICE inline
    void BlockedToWarpStriped(InputT  (&input_items)[ITEMS_PER_THREAD],
                              OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        base_type::blocked_to_warp_striped(input_items, output_items, temp_storage_);
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE inline
    void ScatterToBlocked(InputT  (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD],
                          OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_blocked(input_items, output_items, ranks, temp_storage_);
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE inline
    void ScatterToStriped(InputT  (&input_items)[ITEMS_PER_THREAD],
                          OutputT (&output_items)[ITEMS_PER_THREAD],
                          OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_striped(input_items, output_items, ranks, temp_storage_);
    }

    template<typename OutputT, typename OffsetT>
    HIPCUB_DEVICE inline
    void ScatterToStripedGuarded(InputT  (&input_items)[ITEMS_PER_THREAD],
                                 OutputT (&output_items)[ITEMS_PER_THREAD],
                                 OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_striped_guarded(input_items, output_items, ranks, temp_storage_);
    }

    template<typename OutputT, typename OffsetT, typename ValidFlag>
    HIPCUB_DEVICE inline
    void ScatterToStripedFlagged(InputT    (&input_items)[ITEMS_PER_THREAD],
                                 OutputT   (&output_items)[ITEMS_PER_THREAD],
                                 OffsetT   (&ranks)[ITEMS_PER_THREAD],
                                 ValidFlag (&is_valid)[ITEMS_PER_THREAD])
    {
        base_type::scatter_to_striped_flagged(input_items, output_items, ranks, is_valid, temp_storage_);
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS


    HIPCUB_DEVICE inline void StripedToBlocked(
        InputT      (&items)[ITEMS_PER_THREAD])
    {
        StripedToBlocked(items, items);
    }

    HIPCUB_DEVICE inline void BlockedToStriped(
        InputT      (&items)[ITEMS_PER_THREAD])
    {
        BlockedToStriped(items, items);
    }

    HIPCUB_DEVICE inline void WarpStripedToBlocked(
        InputT      (&items)[ITEMS_PER_THREAD])
    {
        WarpStripedToBlocked(items, items);
    }

    HIPCUB_DEVICE inline void BlockedToWarpStriped(
        InputT      (&items)[ITEMS_PER_THREAD])
    {
        BlockedToWarpStriped(items, items);
    }

    template <typename OffsetT>
    HIPCUB_DEVICE inline void ScatterToBlocked(
        InputT      (&items)[ITEMS_PER_THREAD],
        OffsetT     (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToBlocked(items, items, ranks);
    }

    template <typename OffsetT>
    HIPCUB_DEVICE inline void ScatterToStriped(
        InputT      (&items)[ITEMS_PER_THREAD],
        OffsetT     (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(items, items, ranks);
    }

    template <typename OffsetT>
    HIPCUB_DEVICE inline void ScatterToStripedGuarded(
        InputT      (&items)[ITEMS_PER_THREAD],
        OffsetT     (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStripedGuarded(items, items, ranks);
    }

    template <typename OffsetT, typename ValidFlag>
    HIPCUB_DEVICE inline void ScatterToStripedFlagged(
        InputT      (&items)[ITEMS_PER_THREAD],
        OffsetT     (&ranks)[ITEMS_PER_THREAD],
        ValidFlag   (&is_valid)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(items, items, ranks, is_valid);
    }

#endif

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
