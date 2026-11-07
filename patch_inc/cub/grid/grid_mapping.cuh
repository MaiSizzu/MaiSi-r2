

#ifndef HIPCUB_ROCPRIM_GRID_GRID_MAPPING_HPP_
#define HIPCUB_ROCPRIM_GRID_GRID_MAPPING_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../thread/thread_load.cuh"

BEGIN_HIPCUB_NAMESPACE









enum GridMappingStrategy
{
    
    GRID_MAPPING_RAKE,

    
    GRID_MAPPING_STRIP_MINE,

    
    GRID_MAPPING_DYNAMIC,
};




END_HIPCUB_NAMESPACE

#endif
