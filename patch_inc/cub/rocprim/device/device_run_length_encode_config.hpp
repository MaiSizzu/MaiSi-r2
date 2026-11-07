



















#ifndef ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_
#define ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_CONFIG_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE







template<
    class ReduceByKeyConfig,
    class SelectConfig = default_config
>
struct run_length_encode_config
{

    using reduce_by_key = ReduceByKeyConfig;

    using select = SelectConfig;
};

namespace detail
{

using default_run_length_encode_config = run_length_encode_config<default_config, default_config>;

}

END_ROCPRIM_NAMESPACE




#endif
