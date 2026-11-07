



















#ifndef ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_ADJACENT_DIFFERENCE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"

#include "../block/block_load.hpp"
#include "../block/block_store.hpp"




BEGIN_ROCPRIM_NAMESPACE









template <unsigned int       BlockSize,
          unsigned int       ItemsPerThread,
          block_load_method  LoadMethod  = block_load_method::block_load_transpose,
          block_store_method StoreMethod = block_store_method::block_store_transpose,
          unsigned int       SizeLimit   = ROCPRIM_GRID_SIZE_LIMIT>
struct adjacent_difference_config : kernel_config<BlockSize, ItemsPerThread, SizeLimit>
{
    static constexpr block_load_method  load_method  = LoadMethod;
    static constexpr block_store_method store_method = StoreMethod;
};

namespace detail
{

template <class Value>
struct adjacent_difference_config_fallback
{
    static constexpr unsigned int item_scale
        = ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = adjacent_difference_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

template <unsigned int TargetArch, class Value>
struct default_adjacent_difference_config
    : select_arch<TargetArch, adjacent_difference_config_fallback<Value>>
{
};

}

END_ROCPRIM_NAMESPACE




#endif
