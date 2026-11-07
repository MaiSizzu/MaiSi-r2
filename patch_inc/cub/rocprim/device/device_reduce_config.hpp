



















#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../block/block_reduce.hpp"

#include "config_types.hpp"
#include "detail/device_config_helper.hpp"




BEGIN_ROCPRIM_NAMESPACE


namespace detail
{

template<class Value>
struct reduce_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};

template<class Value>
struct reduce_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};


template<class Value>
struct reduce_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};


template<class Value>
struct reduce_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = reduce_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ::rocprim::block_reduce_algorithm::using_warp_reduce
    >;
};

template<unsigned int TargetArch, class Value>
struct default_reduce_config
    : select_arch<
        TargetArch,
        select_arch_case<803, reduce_config_803<Value>>,
        select_arch_case<900, reduce_config_900<Value>>,
        select_arch_case<ROCPRIM_ARCH_90a, reduce_config_90a<Value>>,
        select_arch_case<1030, reduce_config_1030<Value>>,
        reduce_config_900<Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
