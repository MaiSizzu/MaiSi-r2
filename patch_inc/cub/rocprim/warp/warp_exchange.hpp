



















#ifndef ROCPRIM_WARP_WARP_EXCHANGE_HPP_
#define ROCPRIM_WARP_WARP_EXCHANGE_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../intrinsics/warp_shuffle.hpp"
#include "../functional.hpp"
#include "../types.hpp"




BEGIN_ROCPRIM_NAMESPACE







































template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize = ::rocprim::device_warp_size()
>
class warp_exchange
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");


    struct storage_type_
    {
        T buffer[WarpSize * ItemsPerThread];
    };

public:









    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_;
    #endif



































    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[flat_id * ItemsPerThread + i] = input[i];
        }
        ::rocprim::wave_barrier();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * WarpSize + flat_id];
        }
    }






























    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped_shuffle(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread])
    {
        static_assert(WarpSize % ItemsPerThread == 0,
                      "ItemsPerThread must be a divisor of WarpSize to use blocked_to_striped_shuffle");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        U work_array[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int dst_idx = 0; dst_idx < ItemsPerThread; dst_idx++)
        {
            ROCPRIM_UNROLL
            for(unsigned int src_idx = 0; src_idx < ItemsPerThread; src_idx++)
            {
                const auto value = ::rocprim::warp_shuffle(
                    input[src_idx],
                    flat_id / ItemsPerThread + dst_idx * (WarpSize / ItemsPerThread)
                );
                if(src_idx == flat_id % ItemsPerThread)
                {
                    work_array[dst_idx] = value;
                }
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = work_array[i];
        }
    }



































    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[i * WarpSize + flat_id] = input[i];
        }
        ::rocprim::wave_barrier();

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[flat_id * ItemsPerThread + i];
        }
    }






























    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked_shuffle(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread])
    {
        static_assert(WarpSize % ItemsPerThread == 0,
                      "ItemsPerThread must be a divisor of WarpSize to use striped_to_blocked_shuffle");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        U work_array[ItemsPerThread];

        ROCPRIM_UNROLL
        for(unsigned int dst_idx = 0; dst_idx < ItemsPerThread; dst_idx++)
        {
            ROCPRIM_UNROLL
            for(unsigned int src_idx = 0; src_idx < ItemsPerThread; src_idx++)
            {
                const auto value = ::rocprim::warp_shuffle(
                    input[src_idx],
                    (ItemsPerThread * flat_id + dst_idx) % WarpSize
                );
                if(flat_id / (WarpSize / ItemsPerThread) == src_idx)
                {
                    work_array[dst_idx] = value;
                }
            }
        }

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = work_array[i];
        }
    }








































    template<class U, class OffsetT>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped(
            const T (&input)[ItemsPerThread],
            U (&output)[ItemsPerThread],
            const OffsetT (&ranks)[ItemsPerThread],
            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[ranks[i]] = input[i];
        }
        ::rocprim::wave_barrier();

        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < ItemsPerThread; i++)
        {
            unsigned int item_offset = (i * WarpSize) + flat_id;
            output[i] = storage_.buffer[item_offset];
        }
    }
};

END_ROCPRIM_NAMESPACE




#endif
