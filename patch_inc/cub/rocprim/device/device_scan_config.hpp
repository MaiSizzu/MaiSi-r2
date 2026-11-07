



















#ifndef ROCPRIM_DEVICE_DEVICE_SCAN_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SCAN_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../block/block_load.hpp"
#include "../block/block_store.hpp"
#include "../block/block_scan.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE










template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    bool UseLookback,
    ::rocprim::block_load_method BlockLoadMethod,
    ::rocprim::block_store_method BlockStoreMethod,
    ::rocprim::block_scan_algorithm BlockScanMethod,
    unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct scan_config
{

    static constexpr unsigned int block_size = BlockSize;

    static constexpr unsigned int items_per_thread = ItemsPerThread;

    static constexpr bool use_lookback = UseLookback;

    static constexpr ::rocprim::block_load_method block_load_method = BlockLoadMethod;

    static constexpr ::rocprim::block_store_method block_store_method = BlockStoreMethod;

    static constexpr ::rocprim::block_scan_algorithm block_scan_method = BlockScanMethod;

    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<class Value>
struct scan_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct scan_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};


template<class Value>
struct scan_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 16u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<class Value>
struct scan_config_908
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
        ::rocprim::max(1u, 20u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};


template<class Value>
struct scan_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Value), sizeof(int));

    using type = scan_config<
        limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
        ::rocprim::max(1u, 15u / item_scale),
        ROCPRIM_DETAIL_USE_LOOKBACK_SCAN,
        ::rocprim::block_load_method::block_load_transpose,
        ::rocprim::block_store_method::block_store_transpose,
        ::rocprim::block_scan_algorithm::using_warp_scan
    >;
};

template<unsigned int TargetArch, class Value>
struct default_scan_config
    : select_arch<
        TargetArch,
        select_arch_case<803, scan_config_803<Value>>,
        select_arch_case<900, scan_config_900<Value>>,
        select_arch_case<ROCPRIM_ARCH_90a, scan_config_90a<Value>>,
        select_arch_case<908, scan_config_908<Value>>,
        select_arch_case<1030, scan_config_1030<Value>>,
        scan_config_900<Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
