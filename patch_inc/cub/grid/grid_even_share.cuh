

#ifndef HIPCUB_ROCPRIM_GRID_GRID_EVEN_SHARE_HPP_
#define HIPCUB_ROCPRIM_GRID_GRID_EVEN_SHARE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "grid_mapping.cuh"
#include "../util_type.cuh"

BEGIN_HIPCUB_NAMESPACE





template <typename OffsetT>
struct GridEvenShare
{
private:

    int         total_tiles;
    int         big_shares;
    OffsetT     big_share_items;
    OffsetT     normal_share_items;
    OffsetT     normal_base_offset;

public:


    OffsetT     num_items;


    int         grid_size;


    OffsetT     block_offset;


    OffsetT     block_end;


    OffsetT     block_stride;


    
    __host__ __device__ __forceinline__ GridEvenShare() :
        total_tiles(0),
        big_shares(0),
        big_share_items(0),
        normal_share_items(0),
        normal_base_offset(0),
        num_items(0),
        grid_size(0),
        block_offset(0),
        block_end(0),
        block_stride(0)
    {}


    
    __host__ __device__ __forceinline__ void DispatchInit(
        OffsetT num_items_,
        int     max_grid_size,
        int     tile_items)
    {
        this->block_offset          = num_items_;
        this->block_end             = num_items_;
        this->num_items             = num_items_;
        this->total_tiles           = static_cast<int>(cub::DivideAndRoundUp(num_items_, tile_items));
        this->grid_size             = min(total_tiles, max_grid_size);
        int avg_tiles_per_block     = total_tiles / grid_size;

        this->big_shares            = total_tiles - (avg_tiles_per_block * grid_size);
        this->normal_share_items    = avg_tiles_per_block * tile_items;
        this->normal_base_offset    = big_shares * tile_items;
        this->big_share_items       = normal_share_items + tile_items;
    }


    
    template <int TILE_ITEMS>
    __device__ __forceinline__ void BlockInit(
        int block_id,
        Int2Type<GRID_MAPPING_RAKE> )
    {
        block_stride = TILE_ITEMS;
        if (block_id < big_shares)
        {

            block_offset = (block_id * big_share_items);
            block_end = block_offset + big_share_items;
        }
        else if (block_id < total_tiles)
        {

            block_offset = normal_base_offset + (block_id * normal_share_items);
            block_end = min(num_items, block_offset + normal_share_items);
        }

    }


    
    template <int TILE_ITEMS>
    __device__ __forceinline__ void BlockInit(
        int block_id,
        Int2Type<GRID_MAPPING_STRIP_MINE> )
    {
        block_stride = grid_size * TILE_ITEMS;
        block_offset = (block_id * TILE_ITEMS);
        block_end = num_items;
    }


    
    template <
        int TILE_ITEMS,
        GridMappingStrategy STRATEGY>
    __device__ __forceinline__ void BlockInit()
    {
        BlockInit<TILE_ITEMS>(blockIdx.x, Int2Type<STRATEGY>());
    }


    
    template <int TILE_ITEMS>
    __device__ __forceinline__ void BlockInit(
        OffsetT block_offset,
        OffsetT block_end)
    {
        this->block_offset = block_offset;
        this->block_end = block_end;
        this->block_stride = TILE_ITEMS;
    }


};




END_HIPCUB_NAMESPACE

#endif
