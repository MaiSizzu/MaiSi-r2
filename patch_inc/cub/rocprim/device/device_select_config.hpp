



















#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../block/block_load.hpp"
#include "../block/block_scan.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE










template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    ::rocprim::block_load_method KeyBlockLoadMethod,
    ::rocprim::block_load_method ValueBlockLoadMethod,
    ::rocprim::block_load_method FlagBlockLoadMethod,
    ::rocprim::block_scan_algorithm BlockScanMethod,
    unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct select_config
{

    static constexpr unsigned int block_size = BlockSize;

    static constexpr unsigned int items_per_thread = ItemsPerThread;

    static constexpr block_load_method key_block_load_method = KeyBlockLoadMethod;

    static constexpr block_load_method value_block_load_method = ValueBlockLoadMethod;

    static constexpr block_load_method flag_block_load_method = FlagBlockLoadMethod;

    static constexpr block_scan_algorithm block_scan_method = BlockScanMethod;

    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<class Key>
struct select_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Key), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 13u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Key>
struct select_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Key), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct select_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct select_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = select_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};


template<unsigned int TargetArch, class Key, class >
struct default_select_config
    : select_arch<
        TargetArch,
        select_arch_case<803, select_config_803<Key>>,
        select_arch_case<900, select_config_900<Key>>,
        select_arch_case<ROCPRIM_ARCH_90a, select_config_90a<Key>>,
        select_arch_case<1030, select_config_1030<Key>>,
        select_config_803<Key>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
