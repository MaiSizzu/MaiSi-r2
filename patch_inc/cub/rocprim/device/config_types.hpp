



















#ifndef ROCPRIM_DEVICE_CONFIG_TYPES_HPP_
#define ROCPRIM_DEVICE_CONFIG_TYPES_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../intrinsics/thread.hpp"
#include "../detail/various.hpp"




BEGIN_ROCPRIM_NAMESPACE




struct default_config { };





template <unsigned int BlockSize,
          unsigned int ItemsPerThread,
          unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT>
struct kernel_config
{

    static constexpr unsigned int block_size = BlockSize;

    static constexpr unsigned int items_per_thread = ItemsPerThread;

    static constexpr unsigned int size_limit = SizeLimit;
};

namespace detail
{

template<
    unsigned int MaxBlockSize,
    unsigned int SharedMemoryPerThread,

    unsigned int MinBlockSize, 



    bool = (MaxBlockSize * SharedMemoryPerThread <= (1u << 15))
>
struct limit_block_size
{

    static constexpr unsigned int value =
        limit_block_size<
            detail::next_power_of_two(MaxBlockSize) / 2,
            SharedMemoryPerThread,
            MinBlockSize
        >::value;
};

template<
    unsigned int MaxBlockSize,
    unsigned int SharedMemoryPerThread,
    unsigned int MinBlockSize
>
struct limit_block_size<MaxBlockSize, SharedMemoryPerThread, MinBlockSize, true>
{
    static_assert(MaxBlockSize >= MinBlockSize, "Data is too large, it cannot fit in shared memory");

    static constexpr unsigned int value = MaxBlockSize;
};

template<unsigned int Arch, class T>
struct select_arch_case
{
    static constexpr unsigned int arch = Arch;
    using type = T;
};

template<unsigned int TargetArch, class Case, class... OtherCases>
struct select_arch
    : std::conditional<
        Case::arch == TargetArch,
        extract_type<typename Case::type>,
        select_arch<TargetArch, OtherCases...>
    >::type { };

template<unsigned int TargetArch, class Universal>
struct select_arch<TargetArch, Universal> : extract_type<Universal> { };

template<class Config, class Default>
using default_or_custom_config =
    typename std::conditional<
        std::is_same<Config, default_config>::value,
        Default,
        Config
    >::type;

}

END_ROCPRIM_NAMESPACE




#endif
