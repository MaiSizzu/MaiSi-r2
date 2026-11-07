



















#ifndef ROCPRIM_WARP_DETAIL_WARP_SCAN_SHARED_MEM_HPP_
#define ROCPRIM_WARP_DETAIL_WARP_SCAN_SHARED_MEM_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int WarpSize
>
class warp_scan_shared_mem
{
    struct storage_type_
    {
        T threads[WarpSize];
    };
public:
    using storage_type = detail::raw_storage<storage_type_>;

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output,
                        storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();

        T me = input;
        store_volatile(&storage_.threads[lid], me);
        for(unsigned int i = 1; i < WarpSize; i *= 2)
        {
            if(lid >= i)
            {
                T other = load_volatile(&storage_.threads[lid - i]);
                me = scan_op(other, me);
                store_volatile(&storage_.threads[lid], me);
            }
        }
        output = me;
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input, T& output, T& reduction,
                        storage_type& storage, BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        inclusive_scan(input, output, storage, scan_op);
        reduction = load_volatile(&storage_.threads[WarpSize - 1]);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init,
                        storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        to_exclusive(output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output,
                        storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, output, storage, scan_op);
        to_exclusive(output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input, T& output, T init, T& reduction,
                        storage_type& storage, BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        inclusive_scan(input, output, storage, scan_op);
        reduction = load_volatile(&storage_.threads[WarpSize - 1]);
        to_exclusive(output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T input, T& inclusive_output, T& exclusive_output, T init,
              storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, storage, scan_op);
        to_exclusive(exclusive_output, init, storage, scan_op);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T input, T& inclusive_output, T& exclusive_output,
              storage_type& storage, BinaryFunction scan_op)
    {
        inclusive_scan(input, inclusive_output, storage, scan_op);
        to_exclusive(exclusive_output, storage);
    }

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scan(T input, T& inclusive_output, T& exclusive_output, T init, T& reduction,
              storage_type& storage, BinaryFunction scan_op)
    {
        storage_type_& storage_ = storage.get();
        inclusive_scan(input, inclusive_output, storage, scan_op);
        reduction = load_volatile(&storage_.threads[WarpSize - 1]);
        to_exclusive(exclusive_output, init, storage, scan_op);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T broadcast(T input, const unsigned int src_lane, storage_type& storage)
    {
        storage_type_& storage_ = storage.get();
        if(src_lane == detail::logical_lane_id<WarpSize>())
        {
            store_volatile(&storage_.threads[src_lane], input);
        }
        return load_volatile(&storage_.threads[src_lane]);
    }

protected:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T inclusive_input, T& exclusive_output, storage_type& storage)
    {
        (void) inclusive_input;
        return to_exclusive(exclusive_output, storage);
    }

private:

    template<class BinaryFunction>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T& exclusive_output, T init,
                      storage_type& storage, BinaryFunction scan_op)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();
        exclusive_output = init;
        if(lid != 0)
        {
            exclusive_output = scan_op(init, load_volatile(&storage_.threads[lid-1]));
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_exclusive(T& exclusive_output, storage_type& storage)
    {
        const unsigned int lid = detail::logical_lane_id<WarpSize>();
        storage_type_& storage_ = storage.get();
        if(lid != 0)
        {
            exclusive_output = load_volatile(&storage_.threads[lid-1]);
        }
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
