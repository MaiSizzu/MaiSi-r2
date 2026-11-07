



















#ifndef ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_REDUCE_BY_KEY_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE





template<
    class ScanConfig,
    class ReduceConfig
>
struct reduce_by_key_config
{

    using scan = ScanConfig;

    using reduce = ReduceConfig;
};

namespace detail
{

template<class Key, class Value>
struct reduce_by_key_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using scan = kernel_config<256, 4>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) <= 8 && sizeof(Value) <= 8),
            reduce_by_key_config<scan, kernel_config<256, 7> >
        >,
        reduce_by_key_config<scan, kernel_config<limit_block_size<256U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value, ::rocprim::max(1u, 15u / item_scale)> >
    >;
};

template<class Key, class Value>
struct reduce_by_key_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) <= 8 && sizeof(Value) <= 8),
            reduce_by_key_config<scan, kernel_config<256, 10> >
        >,
        reduce_by_key_config<scan, kernel_config<limit_block_size<256U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value, ::rocprim::max(1u, 15u / item_scale)> >
    >;
};


template<class Key, class Value>
struct reduce_by_key_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) <= 8 && sizeof(Value) <= 8),
            reduce_by_key_config<scan, kernel_config<256, 10> >
        >,
        reduce_by_key_config<scan, kernel_config<limit_block_size<256U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value, ::rocprim::max(1u, 15u / item_scale)> >
    >;
};


template<class Key, class Value>
struct reduce_by_key_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key) + sizeof(Value), 2 * sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) <= 8 && sizeof(Value) <= 8),
            reduce_by_key_config<scan, kernel_config<256, 10> >
        >,
        reduce_by_key_config<scan, kernel_config<limit_block_size<256U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_32>::value, ::rocprim::max(1u, 15u / item_scale)> >
    >;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_reduce_by_key_config
    : select_arch<
        TargetArch,
        select_arch_case<803, reduce_by_key_config_803<Key, Value> >,
        select_arch_case<900, reduce_by_key_config_900<Key, Value> >,
        select_arch_case<ROCPRIM_ARCH_90a, reduce_by_key_config_90a<Key, Value> >,
        select_arch_case<1030, reduce_by_key_config_1030<Key, Value> >,
        reduce_by_key_config_900<Key, Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
