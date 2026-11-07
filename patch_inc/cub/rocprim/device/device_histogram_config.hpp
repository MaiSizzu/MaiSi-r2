



















#ifndef ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_HISTOGRAM_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE








template<
    class HistogramConfig,
    unsigned int MaxGridSize = 1024,
    unsigned int SharedImplMaxBins = 2048
>
struct histogram_config
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using histogram = HistogramConfig;

    static constexpr unsigned int max_grid_size = MaxGridSize;
    static constexpr unsigned int shared_impl_max_bins = SharedImplMaxBins;
#endif
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<
    class HistogramConfig,
    unsigned int MaxGridSize,
    unsigned int SharedImplMaxBins
> constexpr unsigned int
histogram_config<HistogramConfig, MaxGridSize, SharedImplMaxBins>::max_grid_size;
template<
    class HistogramConfig,
    unsigned int MaxGridSize,
    unsigned int SharedImplMaxBins
> constexpr unsigned int
histogram_config<HistogramConfig, MaxGridSize, SharedImplMaxBins>::shared_impl_max_bins;
#endif

namespace detail
{

template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_803
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type = histogram_config<kernel_config<256, ::rocprim::max(10u / Channels / item_scale, 1u)>>;
};

template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_900
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type = histogram_config<kernel_config<256, ::rocprim::max(8u / Channels / item_scale, 1u)>>;
};


template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_90a
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type = histogram_config<kernel_config<256, ::rocprim::max(8u / Channels / item_scale, 1u)>>;
};


template<class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct histogram_config_1030
{
    static constexpr unsigned int item_scale = ::rocprim::detail::ceiling_div(sizeof(Sample), sizeof(int));

    using type = histogram_config<kernel_config<256, ::rocprim::max(8u / Channels / item_scale, 1u)>>;
};

template<unsigned int TargetArch, class Sample, unsigned int Channels, unsigned int ActiveChannels>
struct default_histogram_config
    : select_arch<
        TargetArch,
        select_arch_case<803, histogram_config_803<Sample, Channels, ActiveChannels> >,
        select_arch_case<900, histogram_config_900<Sample, Channels, ActiveChannels> >,
        select_arch_case<ROCPRIM_ARCH_90a, histogram_config_90a<Sample, Channels, ActiveChannels> >,
        select_arch_case<1030, histogram_config_1030<Sample, Channels, ActiveChannels> >,
        histogram_config_900<Sample, Channels, ActiveChannels>
    > { };

}

END_ROCPRIM_NAMESPACE




#endif
