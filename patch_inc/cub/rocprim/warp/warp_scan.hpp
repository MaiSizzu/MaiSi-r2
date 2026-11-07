



















#ifndef ROCPRIM_WARP_WARP_SCAN_HPP_
#define ROCPRIM_WARP_WARP_SCAN_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/warp_scan_crosslane.hpp"
#include "detail/warp_scan_shared_mem.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{


template<class T, unsigned int WarpSize>
struct select_warp_scan_impl
{
    typedef typename std::conditional<

        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_scan_crosslane<T, WarpSize>,
        detail::warp_scan_shared_mem<T, WarpSize>
    >::type type;
};

}


















































template<
    class T,
    unsigned int WarpSize = device_warp_size()
>
class warp_scan
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_warp_scan_impl<T, WarpSize>::type
#endif
{
    using base_type = typename detail::select_warp_scan_impl<T, WarpSize>::type;


    static_assert(WarpSize <= ROCPRIM_MAX_WARP_SIZE, "WarpSize can't be greater than hardware warp size.");

public:








    using storage_type = typename base_type::storage_type;

















































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::inclusive_scan(input, output, storage, scan_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T ,
                        T& ,
                        storage_type& ,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size . Aborting warp sort.");
        return;
    }


















































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::inclusive_scan(input, output, reduction, storage, scan_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto inclusive_scan(T ,
                        T& ,
                        T& ,
                        storage_type& ,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size . Aborting warp sort.");
        return;
    }





















































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T input,
                        T& output,
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::exclusive_scan(input, output, init, storage, scan_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T ,
                        T& ,
                        T ,
                        storage_type& ,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size . Aborting warp sort.");
        return;
    }






















































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto exclusive_scan(T ,
                        T& ,
                        T ,
                        T& ,
                        storage_type& ,
                        BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size . Aborting warp sort.");
        return;
    }



























































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T input,
              T& inclusive_output,
              T& exclusive_output,
              T init,
              storage_type& storage,
              BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::scan(input, inclusive_output, exclusive_output, init, storage, scan_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T ,
              T& ,
              T& ,
              T ,
              storage_type& ,
              BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size . Aborting warp sort.");
        return;
    }



























































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T input,
              T& inclusive_output,
              T& exclusive_output,
              T init,
              T& reduction,
              storage_type& storage,
              BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::scan(
            input, inclusive_output, exclusive_output, init, reduction,
            storage, scan_op
        );
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto scan(T ,
              T& ,
              T& ,
              T ,
              T& ,
              storage_type& ,
              BinaryFunction scan_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) scan_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size . Aborting warp sort.");
        return;
    }










    template<unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto broadcast(T input,
                   const unsigned int src_lane,
                   storage_type& storage)
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), T>::type
    {
        return base_type::broadcast(input, src_lane, storage);
    }



    template<unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto broadcast(T ,
                   const unsigned int ,
                   storage_type& )
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), T>::type
    {
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return T();
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS
protected:

    template<unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto to_exclusive(T inclusive_input, T& exclusive_output, storage_type& storage)
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        return base_type::to_exclusive(inclusive_input, exclusive_output, storage);
    }

    template<unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto to_exclusive(T , T& , storage_type&)
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }
#endif
};

END_ROCPRIM_NAMESPACE




#endif
