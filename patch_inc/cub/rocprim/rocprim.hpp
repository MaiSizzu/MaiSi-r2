



















#ifndef ROCPRIM_HPP_
#define ROCPRIM_HPP_






#include "config.hpp"

#include "rocprim_version.hpp"

#include "intrinsics.hpp"
#include "functional.hpp"
#include "types.hpp"
#include "type_traits.hpp"
#include "iterator.hpp"

#include "warp/warp_reduce.hpp"
#include "warp/warp_scan.hpp"
#include "warp/warp_sort.hpp"

#include "block/block_discontinuity.hpp"
#include "block/block_exchange.hpp"
#include "block/block_histogram.hpp"
#include "block/block_load.hpp"
#include "block/block_radix_sort.hpp"
#include "block/block_scan.hpp"
#include "block/block_sort.hpp"
#include "block/block_store.hpp"

#include "device/device_adjacent_difference.hpp"
#include "device/device_binary_search.hpp"
#include "device/device_histogram.hpp"
#include "device/device_merge.hpp"
#include "device/device_merge_sort.hpp"
#include "device/device_partition.hpp"
#include "device/device_radix_sort.hpp"
#include "device/device_reduce_by_key.hpp"
#include "device/device_reduce.hpp"
#include "device/device_run_length_encode.hpp"
#include "device/device_scan_by_key.hpp"
#include "device/device_scan.hpp"
#include "device/device_segmented_radix_sort.hpp"
#include "device/device_segmented_reduce.hpp"
#include "device/device_segmented_scan.hpp"
#include "device/device_select.hpp"
#include "device/device_transform.hpp"

BEGIN_ROCPRIM_NAMESPACE



ROCPRIM_HOST_DEVICE inline
unsigned int version()
{
    return ROCPRIM_VERSION;
}

END_ROCPRIM_NAMESPACE

#endif
