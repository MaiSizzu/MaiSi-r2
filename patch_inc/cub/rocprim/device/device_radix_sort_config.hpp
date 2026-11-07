



















#ifndef ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_RADIX_SORT_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE















template<
    unsigned int LongRadixBits,
    unsigned int ShortRadixBits,
    class ScanConfig,
    class SortConfig,
    class SortSingleConfig = kernel_config<256, 10>,
    class SortMergeConfig = kernel_config<1024, 1>,
    unsigned int MergeSizeLimitBlocks = 1024U,
    bool ForceSingleKernelConfig = false
>
struct radix_sort_config
{

    static constexpr unsigned int long_radix_bits = LongRadixBits;

    static constexpr unsigned int short_radix_bits = ShortRadixBits;

    static constexpr unsigned int merge_size_limit_blocks = MergeSizeLimitBlocks;


    using scan = ScanConfig;

    using sort = SortConfig;

    using sort_single = SortSingleConfig;

    using sort_merge = SortMergeConfig;

    static constexpr bool force_single_kernel_config = ForceSingleKernelConfig;
};

namespace detail
{

template<class Key, class Value>
struct radix_sort_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<
                8, 7, scan,
                kernel_config<256, 10>, kernel_config<256, 19>
            >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<
                8, 7, scan,
                kernel_config<256, 10>, kernel_config<256, 17>
            >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<
                7, 6, scan,
                kernel_config<256, 15>, kernel_config<256, 13>
            >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<
                7, 6, scan,
                kernel_config<256, 13>, kernel_config<256, 10>
            >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_803<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<8, 7, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<8, 7, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 16> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 9>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 7>, kernel_config<256, 12> > >
    > { };

template<class Key, class Value>
struct radix_sort_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 10>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 10>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 12> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_900<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 16> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 17>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 15>, kernel_config<256, 12> > >
    > { };


template<class Key, class Value>
struct radix_sort_config_908
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 10>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 10>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, kernel_config<256, 4>,
            kernel_config<256, 15>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, kernel_config<256, 4>,
            kernel_config<256, 14>, kernel_config<256, 12> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_908<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 17> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 4>, kernel_config<256, 17>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 4>, kernel_config<256, 15>, kernel_config<256, 12> > >
    > { };


template<class Key, class Value>
struct radix_sort_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 1>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 5>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 5>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 7>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 7>, kernel_config<256, 14> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_90a<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 1>, kernel_config<256, 5>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 1>, kernel_config<256, 5>, kernel_config<256, 17> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 1>, kernel_config<256, 8>, kernel_config<256, 15> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 1>, kernel_config<256, 7>, kernel_config<256, 14> > >
    > { };


template<class Key, class Value>
struct radix_sort_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));

    using scan = kernel_config<256, 2>;

    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            radix_sort_config<4, 4, scan,
            kernel_config<256, 10>, kernel_config<256, 19> >
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            radix_sort_config<6, 5, scan,
            kernel_config<256, 10>, kernel_config<256, 17> >
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 15> >
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            radix_sort_config<7, 6, scan,
            kernel_config<256, 15>, kernel_config<256, 14> >
        >,
        radix_sort_config<
            6, 4, scan,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_32>::value,
                ::rocprim::max(1u, 15u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >,
            kernel_config<
                limit_block_size<256U, sizeof(Value), ROCPRIM_WARP_SIZE_64>::value,
                ::rocprim::max(1u, 10u / item_scale)
            >
        >
    >;
};

template<class Key>
struct radix_sort_config_1030<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, radix_sort_config<4, 3, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 2, radix_sort_config<6, 5, kernel_config<256, 2>, kernel_config<256, 10>, kernel_config<256, 19> > >,
        select_type_case<sizeof(Key) == 4, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 17>, kernel_config<256, 17> > >,
        select_type_case<sizeof(Key) == 8, radix_sort_config<7, 6, kernel_config<256, 2>, kernel_config<256, 15>, kernel_config<256, 15> > >
    > { };

template<unsigned int TargetArch, class Key, class Value>
struct default_radix_sort_config
    : select_arch<
        TargetArch,
        select_arch_case<803, radix_sort_config_803<Key, Value> >,
        select_arch_case<900, radix_sort_config_900<Key, Value> >,
        select_arch_case<908, radix_sort_config_908<Key, Value> >,
        select_arch_case<ROCPRIM_ARCH_90a, radix_sort_config_90a<Key, Value> >,
        select_arch_case<1030, radix_sort_config_1030<Key, Value> >,
        radix_sort_config_900<Key, Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
