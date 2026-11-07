



















#ifndef ROCPRIM_WARP_WARP_REDUCE_HPP_
#define ROCPRIM_WARP_WARP_REDUCE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "detail/warp_reduce_crosslane.hpp"
#include "detail/warp_reduce_shared_mem.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{


template<class T, unsigned int WarpSize, bool UseAllReduce>
struct select_warp_reduce_impl
{
    typedef typename std::conditional<

        detail::is_warpsize_shuffleable<WarpSize>::value,
        detail::warp_reduce_crosslane<T, WarpSize, UseAllReduce>,
        detail::warp_reduce_shared_mem<T, WarpSize, UseAllReduce>
    >::type type;
};

}




















































template<
    class T,
    unsigned int WarpSize = device_warp_size(),
    bool UseAllReduce = false
>
class warp_reduce
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_warp_reduce_impl<T, WarpSize, UseAllReduce>::type
#endif
{
    using base_type = typename detail::select_warp_reduce_impl<T, WarpSize, UseAllReduce>::type;


    static_assert(WarpSize <= ROCPRIM_MAX_WARP_SIZE, "WarpSize can't be greater than hardware warp size.");

public:








    using storage_type = typename base_type::storage_type;













































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::reduce(input, output, storage, reduce_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto reduce(T ,
                T& ,
                storage_type& ,
                BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) reduce_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }















































    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto reduce(T input,
                T& output,
                int valid_items,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::reduce(input, output, valid_items, storage, reduce_op);
    }



    template<class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto reduce(T ,
                T& ,
                int ,
                storage_type& ,
                BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) reduce_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }



















    template<class Flag, class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto head_segmented_reduce(T input,
                               T& output,
                               Flag flag,
                               storage_type& storage,
                               BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::head_segmented_reduce(input, output, flag, storage, reduce_op);
    }



    template<class Flag, class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto head_segmented_reduce(T ,
                               T& ,
                               Flag ,
                               storage_type& ,
                               BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) reduce_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }



















    template<class Flag, class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto tail_segmented_reduce(T input,
                               T& output,
                               Flag flag,
                               storage_type& storage,
                               BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::tail_segmented_reduce(input, output, flag, storage, reduce_op);
    }



    template<class Flag, class BinaryFunction = ::rocprim::plus<T>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto tail_segmented_reduce(T ,
                               T& ,
                               Flag ,
                               storage_type& ,
                               BinaryFunction reduce_op = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) reduce_op;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }
};

END_ROCPRIM_NAMESPACE




#endif
