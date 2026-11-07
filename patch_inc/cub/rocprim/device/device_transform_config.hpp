



















#ifndef ROCPRIM_DEVICE_DEVICE_TRANSFORM_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_TRANSFORM_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../functional.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE


template <unsigned int BlockSize,
          unsigned int ItemsPerThread,
          unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
using transform_config = kernel_config<BlockSize, ItemsPerThread, SizeLimit>;

namespace detail
{

template<class Value>
struct transform_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = transform_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

template<class Value>
struct transform_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = transform_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

template<class Value>
struct transform_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = transform_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

template<class Value>
struct transform_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = transform_config<256, ::rocprim::max(1u, 16u / item_scale)>;
};

template<unsigned int TargetArch, class Value>
struct default_transform_config
    : select_arch<
        TargetArch,
        select_arch_case<803, transform_config_803<Value>>,
        select_arch_case<900, transform_config_900<Value>>,
        select_arch_case<ROCPRIM_ARCH_90a, transform_config_90a<Value>>,
        select_arch_case<1030, transform_config_1030<Value>>,
        transform_config_900<Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
