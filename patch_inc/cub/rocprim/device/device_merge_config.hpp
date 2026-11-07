



















#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE


template<unsigned int BlockSize, unsigned int ItemsPerThread>
using merge_config = kernel_config<BlockSize, ItemsPerThread>;

namespace detail
{

template<class Key, class Value>
struct merge_config_803
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));


    using type = merge_config<256, ::rocprim::max(1u, 10u / item_scale)>;
};

template<class Key>
struct merge_config_803<Key, empty_type>
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_type<
        select_type_case<sizeof(Key) <= 2, merge_config<256, 11> >,
        select_type_case<sizeof(Key) <= 4, merge_config<256, 10> >,
        select_type_case<sizeof(Key) <= 8, merge_config<256, 7> >,
        merge_config<256, ::rocprim::max(1u, 10u / item_scale)>
    >;
};

template<class Key, class Value>
struct merge_config_900
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));


    using type = merge_config<256, ::rocprim::max(1u, 10u / item_scale)>;
};

template<class Key>
struct merge_config_900<Key, empty_type>
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_type<
        select_type_case<sizeof(Key) <= 2, merge_config<256, 11> >,
        select_type_case<sizeof(Key) <= 4, merge_config<256, 10> >,
        select_type_case<sizeof(Key) <= 8, merge_config<256, 7> >,
        merge_config<256, ::rocprim::max(1u, 10u / item_scale)>
    >;
};


template<class Key, class Value>
struct merge_config_90a
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));


    using type = merge_config<256, ::rocprim::max(1u, 10u / item_scale)>;
};

template<class Key>
struct merge_config_90a<Key, empty_type>
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_type<
        select_type_case<sizeof(Key) <= 2, merge_config<256, 11> >,
        select_type_case<sizeof(Key) <= 4, merge_config<256, 10> >,
        select_type_case<sizeof(Key) <= 8, merge_config<256, 7> >,
        merge_config<256, ::rocprim::max(1u, 10u / item_scale)>
    >;
};


template<class Key, class Value>
struct merge_config_1030
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(::rocprim::max(sizeof(Key), sizeof(Value)), sizeof(int));


    using type = merge_config<256, ::rocprim::max(1u, 10u / item_scale)>;
};

template<class Key>
struct merge_config_1030<Key, empty_type>
{
    static constexpr unsigned int item_scale =
        ::rocprim::detail::ceiling_div<unsigned int>(sizeof(Key), sizeof(int));

    using type = select_type<
        select_type_case<sizeof(Key) <= 2, merge_config<256, 11> >,
        select_type_case<sizeof(Key) <= 4, merge_config<256, 10> >,
        select_type_case<sizeof(Key) <= 8, merge_config<256, 7> >,
        merge_config<256, ::rocprim::max(1u, 10u / item_scale)>
    >;
};

template<unsigned int TargetArch, class Key, class Value>
struct default_merge_config
    : select_arch<
        TargetArch,
        select_arch_case<803, merge_config_803<Key, Value>>,
        select_arch_case<900, merge_config_900<Key, Value>>,
        select_arch_case<ROCPRIM_ARCH_90a, merge_config_90a<Key, Value>>,
        select_arch_case<1030, merge_config_1030<Key, Value>>,
        merge_config_900<Key, Value>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
