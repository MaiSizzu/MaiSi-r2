



















#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_SORT_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../functional.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    template <unsigned int SortBlockSize,
              unsigned int SortItemsPerThread,
              unsigned int MergeImpl1BlockSize,
              unsigned int MergeImplMPPartitionBlockSize,
              unsigned int MergeImplMPBlockSize,
              unsigned int MergeImplMPItemsPerThread,
              unsigned int MinInputSizeMergepath>
    struct merge_sort_config_impl
    {
        using sort_config                      = kernel_config<SortBlockSize, SortItemsPerThread>;
        using merge_impl1_config               = kernel_config<MergeImpl1BlockSize, 1>;
        using merge_mergepath_partition_config = kernel_config<MergeImplMPPartitionBlockSize, 1>;
        using merge_mergepath_config
            = kernel_config<MergeImplMPBlockSize, MergeImplMPItemsPerThread>;
        static constexpr unsigned int min_input_size_mergepath = MinInputSizeMergepath;
    };
}










template<unsigned int     MergeImpl1BlockSize           = 512,
         unsigned int     SortBlockSize                 = MergeImpl1BlockSize,
         unsigned int     SortItemsPerThread            = 1,
         unsigned int     MergeImplMPPartitionBlockSize = 128,
         unsigned int     MergeImplMPBlockSize          = std::min(SortBlockSize, 128u),
         unsigned int     MergeImplMPItemsPerThread
         = SortBlockSize* SortItemsPerThread / MergeImplMPBlockSize,
         unsigned int     MinInputSizeMergepath = 200000>
using merge_sort_config = detail::merge_sort_config_impl<SortBlockSize,
                                                         SortItemsPerThread,
                                                         MergeImpl1BlockSize,
                                                         MergeImplMPPartitionBlockSize,
                                                         MergeImplMPBlockSize,
                                                         MergeImplMPItemsPerThread,
                                                         MinInputSizeMergepath>;

namespace detail
{

template<class Key, class Value>
struct merge_sort_config_803
{
    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            merge_sort_config<64U>
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            merge_sort_config<256U>
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            merge_sort_config<512U>
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            merge_sort_config<1024U>
        >,
        merge_sort_config<limit_block_size<1024U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value>
    >;
};

template<class Value>
struct merge_sort_config_803<rocprim::half, Value>
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(rocprim::half) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value>;
};

template<class Key>
struct merge_sort_config_803<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, merge_sort_config<64U> >,
        select_type_case<sizeof(Key) == 2, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) == 4, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) >= 8, merge_sort_config<limit_block_size<512U, sizeof(Key), ROCPRIM_WARP_SIZE_64>::value> >
    > { };

template<>
struct merge_sort_config_803<rocprim::half, empty_type>
{
    using type = merge_sort_config<256U>;
};

template<class Key, class Value, bool = is_scalar<Key>::value>
struct merge_sort_config_900
{
    using type = select_type<

        select_type_case<(sizeof(Key) == 1 && sizeof(Value) <= 16), merge_sort_config<512U, 512U, 2U>>,
        select_type_case<(sizeof(Key) == 2 && sizeof(Value) <= 16), merge_sort_config<512U, 256U, 4U>>,
        select_type_case<(sizeof(Key) == 4 && sizeof(Value) <= 16), merge_sort_config<512U, 256U, 4U>>,
        select_type_case<(sizeof(Key) == 8 && sizeof(Value) <= 16), merge_sort_config<256U, 256U, 4U>>,

        merge_sort_config<
            limit_block_size<1024U,
                             ::rocprim::max(sizeof(Key) + sizeof(unsigned int), sizeof(Value)),
                             ROCPRIM_WARP_SIZE_64>::value>>;
};

template<class Key, class Value>
struct merge_sort_config_900<Key, Value, false>
{
    using type = select_type<

        select_type_case<(sizeof(Key) == 8  && sizeof(Value) <= 16), merge_sort_config<512U, 512U, 2U>>,
        select_type_case<(sizeof(Key) == 16 && sizeof(Value) <= 16), merge_sort_config<512U, 512U, 2U>>,

        merge_sort_config<
            limit_block_size<512U,
                             ::rocprim::max(sizeof(Key) + sizeof(unsigned int), sizeof(Value)),
                             ROCPRIM_WARP_SIZE_64>::value>>;
};


template<class Key, class Value>
struct merge_sort_config_1030
{
    using type = select_type<
        select_type_case<
            (sizeof(Key) == 1 && sizeof(Value) <= 8),
            merge_sort_config<64U>
        >,
        select_type_case<
            (sizeof(Key) == 2 && sizeof(Value) <= 8),
            merge_sort_config<256U>
        >,
        select_type_case<
            (sizeof(Key) == 4 && sizeof(Value) <= 8),
            merge_sort_config<512U>
        >,
        select_type_case<
            (sizeof(Key) == 8 && sizeof(Value) <= 8),
            merge_sort_config<1024U>
        >,
        merge_sort_config<limit_block_size<1024U, sizeof(Key) + sizeof(Value), ROCPRIM_WARP_SIZE_64>::value>
    >;
};

template<class Value>
struct merge_sort_config_1030<rocprim::half, Value>
{
    using type = merge_sort_config<limit_block_size<256U, sizeof(rocprim::half) + sizeof(Value), ROCPRIM_WARP_SIZE_32>::value>;
};

template<class Key>
struct merge_sort_config_1030<Key, empty_type>
    : select_type<
        select_type_case<sizeof(Key) == 1, merge_sort_config<64U> >,
        select_type_case<sizeof(Key) == 2, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) == 4, merge_sort_config<256U> >,
        select_type_case<sizeof(Key) >= 8, merge_sort_config<limit_block_size<512U, sizeof(Key), ROCPRIM_WARP_SIZE_32>::value> >
    > { };

template<>
struct merge_sort_config_1030<rocprim::half, empty_type>
{
    using type = merge_sort_config<256U>;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_merge_sort_config
    : select_arch<
        TargetArch,
        select_arch_case<803, merge_sort_config_803<Key, Value>>,
        select_arch_case<900, merge_sort_config_900<Key, Value>>,
        select_arch_case<1030, merge_sort_config_1030<Key, Value>>,
        merge_sort_config_900<Key, Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
