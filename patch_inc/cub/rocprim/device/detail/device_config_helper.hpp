



















#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_HELPER_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_HELPER_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../block/block_reduce.hpp"

#include "../config_types.hpp"




BEGIN_ROCPRIM_NAMESPACE







template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    ::rocprim::block_reduce_algorithm BlockReduceMethod,
    unsigned int SizeLimit = ROCPRIM_GRID_SIZE_LIMIT
>
struct reduce_config
{

    static constexpr unsigned int block_size = BlockSize;

    static constexpr unsigned int items_per_thread = ItemsPerThread;

    static constexpr block_reduce_algorithm block_reduce_method = BlockReduceMethod;

    static constexpr unsigned int size_limit = SizeLimit;
};

END_ROCPRIM_NAMESPACE

#endif
