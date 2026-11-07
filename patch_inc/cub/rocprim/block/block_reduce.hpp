



















#ifndef ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_
#define ROCPRIM_BLOCK_BLOCK_REDUCE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_reduce_warp_reduce.hpp"
#include "detail/block_reduce_raking_reduce.hpp"





BEGIN_ROCPRIM_NAMESPACE


enum class block_reduce_algorithm
{

    using_warp_reduce,

    raking_reduce,

    raking_reduce_commutative_only,

    default_algorithm = using_warp_reduce,
};

namespace detail
{



template<block_reduce_algorithm Algorithm>
struct select_block_reduce_impl;

template<>
struct select_block_reduce_impl<block_reduce_algorithm::using_warp_reduce>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
    using type = block_reduce_warp_reduce<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
};

template<>
struct select_block_reduce_impl<block_reduce_algorithm::raking_reduce>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
    using type = block_reduce_raking_reduce<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
};

template<>
struct select_block_reduce_impl<block_reduce_algorithm::raking_reduce_commutative_only>
{
    template<class T, unsigned int BlockSizeX, unsigned int BlockSizeY, unsigned int BlockSizeZ>
    using type = block_reduce_raking_reduce<T, BlockSizeX, BlockSizeY, BlockSizeZ, true>;
};


}















































template<
    class T,
    unsigned int BlockSizeX,
    block_reduce_algorithm Algorithm = block_reduce_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_reduce
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_reduce_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ>
#endif
{
    using base_type = typename detail::select_block_reduce_impl<Algorithm>::template type<T, BlockSizeX, BlockSizeY, BlockSizeZ>;
public:








    using storage_type = typename base_type::storage_type;















































    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input,
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, storage, reduce_op);
    }
















    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void reduce(T input,
                T& output,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, reduce_op);
    }
















































    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, storage, reduce_op);
    }

















    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::plus<T>
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void reduce(T (&input)[ItemsPerThread],
                T& output,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, reduce_op);
    }















































    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                storage_type& storage,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, valid_items, storage, reduce_op);
    }


















    template<class BinaryFunction = ::rocprim::plus<T>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void reduce(T input,
                T& output,
                unsigned int valid_items,
                BinaryFunction reduce_op = BinaryFunction())
    {
        base_type::reduce(input, output, valid_items, reduce_op);
    }
};

END_ROCPRIM_NAMESPACE




#endif
