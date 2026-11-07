



















#ifndef ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_SEGMENTED_RADIX_SORT_CONFIG_HPP_

#include <algorithm>
#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE



















template<unsigned int LogicalWarpSizeSmall,
         unsigned int ItemsPerThreadSmall,
         unsigned int BlockSizeSmall              = 256,
         unsigned int PartitioningThreshold       = 3000,
         bool         EnableUnpartitionedWarpSort = true,
         unsigned int LogicalWarpSizeMedium       = std::max(32u, LogicalWarpSizeSmall),
         unsigned int ItemsPerThreadMedium        = std::max(4u, ItemsPerThreadSmall),
         unsigned int BlockSizeMedium             = 256>
struct WarpSortConfig
{
    static_assert(LogicalWarpSizeSmall * ItemsPerThreadSmall
                      <= LogicalWarpSizeMedium * ItemsPerThreadMedium,
                  "The number of items processed by a small warp cannot be larger than the number "
                  "of items processed by a medium warp");

    static constexpr unsigned int logical_warp_size_small = LogicalWarpSizeSmall;

    static constexpr unsigned int items_per_thread_small = ItemsPerThreadSmall;

    static constexpr unsigned int block_size_small = BlockSizeSmall;


    static constexpr unsigned int partitioning_threshold = PartitioningThreshold;


    static constexpr bool enable_unpartitioned_warp_sort = EnableUnpartitionedWarpSort;

    static constexpr unsigned int logical_warp_size_medium = LogicalWarpSizeMedium;

    static constexpr unsigned int items_per_thread_medium = ItemsPerThreadMedium;

    static constexpr unsigned int block_size_medium = BlockSizeMedium;
};



struct DisabledWarpSortConfig
{

    static constexpr unsigned int logical_warp_size_small = 1;

    static constexpr unsigned int items_per_thread_small = 1;

    static constexpr unsigned int block_size_small = 1;


    static constexpr unsigned int partitioning_threshold = 0;


    static constexpr bool enable_unpartitioned_warp_sort = false;

    static constexpr unsigned int logical_warp_size_medium = 1;

    static constexpr unsigned int items_per_thread_medium = 1;

    static constexpr unsigned int block_size_medium = 1;
};





template<class Key, unsigned int MediumWarpSize = ROCPRIM_WARP_SIZE_32>
using select_warp_sort_config_t
    = std::conditional_t<sizeof(Key) < 2,
                         DisabledWarpSortConfig,
                         WarpSortConfig<32,
                                        4,
                                        256,
                                        3000,
                                        (sizeof(Key) > 2),
                                        MediumWarpSize,
                                        4,
                                        256
                                        >>;

















template<
    unsigned int LongRadixBits,
    unsigned int ShortRadixBits,
    class SortConfig,
    class WarpSortConfig = DisabledWarpSortConfig
>
struct segmented_radix_sort_config
{

    static constexpr unsigned int long_radix_bits = LongRadixBits;

    static constexpr unsigned int short_radix_bits = ShortRadixBits;

    using sort = SortConfig;

    using warp_sort_config = WarpSortConfig;
};

namespace detail
{

template<class Key, class Value>
struct segmented_radix_sort_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 13>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_803<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<8, 7, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 9>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 7>, select_warp_sort_config_t<Key> > >
    > { };

template<class Key, class Value>
struct segmented_radix_sort_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<4, 4, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_900<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<4, 3, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 17>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> > >
    > { };

template<class Key, class Value>
struct segmented_radix_sort_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<4,
                                        4,
                                        kernel_config<256, 10>,
                                        select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<6,
                                        5,
                                        kernel_config<256, 10>,
                                        select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7,
                                        6,
                                        kernel_config<256, 15>,
                                        select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7,
                                        6,
                                        kernel_config<256, 15>,
                                        select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
        segmented_radix_sort_config<7,
                                    6,
                                    kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>,
                                    select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>;
};

template<class Key>
struct segmented_radix_sort_config_90a<Key, empty_type>
    : select_type<
          select_type_case<
              sizeof(Key) == 1,
              segmented_radix_sort_config<4,
                                          3,
                                          kernel_config<256, 10>,
                                          select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
          select_type_case<
              sizeof(Key) == 2,
              segmented_radix_sort_config<6,
                                          5,
                                          kernel_config<256, 10>,
                                          select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
          select_type_case<
              sizeof(Key) == 4,
              segmented_radix_sort_config<7,
                                          6,
                                          kernel_config<256, 17>,
                                          select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>,
          select_type_case<
              sizeof(Key) == 8,
              segmented_radix_sort_config<7,
                                          6,
                                          kernel_config<256, 15>,
                                          select_warp_sort_config_t<Key, ROCPRIM_WARP_SIZE_64>>>>
{};

template<class Key, class Value>
struct segmented_radix_sort_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            segmented_radix_sort_config<4, 4, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> >
        >,
        segmented_radix_sort_config<7, 6, kernel_config<256, ::rocprim::max(1u, 15u / item_scale)>, select_warp_sort_config_t<Key> >
    >;
};

template<class Key>
struct segmented_radix_sort_config_1030<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, segmented_radix_sort_config<4, 3, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 2, segmented_radix_sort_config<6, 5, kernel_config<256, 10>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 4, segmented_radix_sort_config<7, 6, kernel_config<256, 17>, select_warp_sort_config_t<Key> > >,
        select_type_case<sizeof(Key) == 8, segmented_radix_sort_config<7, 6, kernel_config<256, 15>, select_warp_sort_config_t<Key> > >
    > { };

template<unsigned int TargetArch, class Key, class Value>
struct default_segmented_radix_sort_config
    : select_arch<
          TargetArch,
          select_arch_case<803, detail::segmented_radix_sort_config_803<Key, Value>>,
          select_arch_case<900, detail::segmented_radix_sort_config_900<Key, Value>>,
          select_arch_case<906, detail::segmented_radix_sort_config_90a<Key, Value>>,
          select_arch_case<908, detail::segmented_radix_sort_config_90a<Key, Value>>,
          select_arch_case<ROCPRIM_ARCH_90a, detail::segmented_radix_sort_config_90a<Key, Value>>,
          select_arch_case<1030, detail::segmented_radix_sort_config_1030<Key, Value>>,
          detail::segmented_radix_sort_config_900<Key, Value>>
{};

}

END_ROCPRIM_NAMESPACE




#endif
