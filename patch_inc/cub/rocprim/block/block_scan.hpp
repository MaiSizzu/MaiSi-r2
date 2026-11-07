



















#ifndef ROCPRIM_BLOCK_BLOCK_SCAN_HPP_
#define ROCPRIM_BLOCK_BLOCK_SCAN_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_scan_warp_scan.hpp"
#include "detail/block_scan_reduce_then_scan.hpp"




BEGIN_ROCPRIM_NAMESPACE


enum class block_scan_algorithm
{

    using_warp_scan,

    reduce_then_scan,

    default_algorithm = using_warp_scan,
};

namespace detail
{



template<block_scan_algorithm Algorithm>
struct select_block_scan_impl;

template<>
struct select_block_scan_impl<block_scan_algorithm::using_warp_scan>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
    using type = block_scan_warp_scan<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
};

template<>
struct select_block_scan_impl<block_scan_algorithm::reduce_then_scan>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>


    using type = typename std::conditional<
                    (BlockSizeX * BlockSizeY * BlockSizeZ <= ::rocprim::device_warp_size()),
                    block_scan_warp_scan<T, BlockSizeX, BlockSizeY, BlockSizeZ>,
                    block_scan_reduce_then_scan<T, BlockSizeX, BlockSizeY, BlockSizeZ>
                 >::type;
};

}















































template<
    class T,
    unsigned int BlockSizeX,
    block_scan_algorithm Algorithm = block_scan_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_scan
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_scan_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ>
#endif
{
    using base_type = typename detail::select_block_scan_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
public:








    using storage_type = typename base_type::storage_type;















































    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, storage, scan_op);
    }
















    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T input,
                        T& output,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, scan_op);
    }



















































    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, reduction, storage, scan_op);
    }

















    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T input,
                        T& output,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::inclusive_scan(input, output, reduction, scan_op);
    }













































































    template<
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        base_type::inclusive_scan(input, output, storage, prefix_callback_op, scan_op);
    }
















































    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], storage, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, storage, scan_op);
        }
    }

















    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, scan_op);
        }
    }



















































    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], reduction, storage, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, reduction, storage, scan_op);
        }
    }


















    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], reduction, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, reduction, scan_op);
        }
    }














































































    template<
        unsigned int ItemsPerThread,
        class PrefixCallback,
        class BinaryFunction
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void inclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        if(ItemsPerThread == 1)
        {
            base_type::inclusive_scan(input[0], output[0], storage, prefix_callback_op, scan_op);
        }
        else
        {
            base_type::inclusive_scan(input, output, storage, prefix_callback_op, scan_op);
        }
    }



















































    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, storage, scan_op);
    }


















    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, scan_op);
    }























































    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
    }



















    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T input,
                        T& output,
                        T init,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        base_type::exclusive_scan(input, output, init, reduction, scan_op);
    }













































































    template<
        class PrefixCallback,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T input,
                        T& output,
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        base_type::exclusive_scan(input, output, storage, prefix_callback_op, scan_op);
    }




















































    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, storage, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, storage, scan_op);
        }
    }



















    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, scan_op);
        }
    }
























































    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        storage_type& storage,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, reduction, storage, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, reduction, storage, scan_op);
        }
    }




















    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        T init,
                        T& reduction,
                        BinaryFunction scan_op = BinaryFunction())
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], init, reduction, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, init, reduction, scan_op);
        }
    }














































































    template<
        unsigned int ItemsPerThread,
        class PrefixCallback,
        class BinaryFunction
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(T (&input)[ItemsPerThread],
                        T (&output)[ItemsPerThread],
                        storage_type& storage,
                        PrefixCallback& prefix_callback_op,
                        BinaryFunction scan_op)
    {
        if(ItemsPerThread == 1)
        {
            base_type::exclusive_scan(input[0], output[0], storage, prefix_callback_op, scan_op);
        }
        else
        {
            base_type::exclusive_scan(input, output, storage, prefix_callback_op, scan_op);
        }
    }
};

END_ROCPRIM_NAMESPACE




#endif
