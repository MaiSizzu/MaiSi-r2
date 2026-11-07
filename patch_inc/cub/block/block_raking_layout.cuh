




#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RAKING_LAYOUT_HPP_

#include <type_traits>

#include "../config.hpp"

#include <cub/rocprim/config.hpp>
#include <cub/rocprim/detail/various.hpp>

BEGIN_HIPCUB_NAMESPACE


template <
    typename    T,
    int         BLOCK_THREADS,
    int ARCH = HIPCUB_ARCH 
>
struct block_raking_layout
{




    enum
    {

        SHARED_ELEMENTS = BLOCK_THREADS,


        MAX_RAKING_THREADS = ::rocprim::detail::get_min_warp_size(BLOCK_THREADS, HIPCUB_DEVICE_WARP_THREADS),


        SEGMENT_LENGTH = (SHARED_ELEMENTS + MAX_RAKING_THREADS - 1) / MAX_RAKING_THREADS,


        RAKING_THREADS = (SHARED_ELEMENTS + SEGMENT_LENGTH - 1) / SEGMENT_LENGTH,


        USE_SEGMENT_PADDING = ((SEGMENT_LENGTH & 1) == 0) && (SEGMENT_LENGTH > 2),


        GRID_ELEMENTS = RAKING_THREADS * (SEGMENT_LENGTH + USE_SEGMENT_PADDING),


        UNGUARDED = (SHARED_ELEMENTS % RAKING_THREADS == 0),
    };


#ifndef DOXYGEN_SHOULD_SKIP_THIS
    
    struct __align__(16) _TempStorage
    {
        T buff[BlockRakingLayout::GRID_ELEMENTS];
    };

#endif


    struct TempStorage : Uninitialized<_TempStorage> {};

    
    static HIPCUB_DEVICE inline T* PlacementPtr(
        TempStorage &temp_storage,
        unsigned int linear_tid)
    {

        unsigned int offset = linear_tid;


        if (USE_SEGMENT_PADDING > 0)
        {
            offset += offset / SEGMENT_LENGTH;
        }


        return temp_storage.Alias().buff + offset;
    }

    
    static HIPCUB_DEVICE inline T* RakingPtr(
        TempStorage &temp_storage,
        unsigned int linear_tid)
    {
        return temp_storage.Alias().buff + (linear_tid * (SEGMENT_LENGTH + USE_SEGMENT_PADDING));
    }
};

END_HIPCUB_NAMESPACE

#endif
