



















#ifndef ROCPRIM_BLOCK_BLOCK_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/block_sort_bitonic.hpp"




BEGIN_ROCPRIM_NAMESPACE


enum class block_sort_algorithm
{

    bitonic_sort,

    default_algorithm = bitonic_sort,
};

namespace detail
{



template<block_sort_algorithm Algorithm>
struct select_block_sort_impl;

template<>
struct select_block_sort_impl<block_sort_algorithm::bitonic_sort>
{
    template <class Key,
              unsigned int BlockSizeX,
              unsigned int BlockSizeY,
              unsigned int BlockSizeZ,
              unsigned int ItemsPerThread,
              class Value>
    using type = block_sort_bitonic<Key, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Value>;
};

}












































template<
    class Key,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread = 1,
    class Value = empty_type,
    block_sort_algorithm Algorithm = block_sort_algorithm::default_algorithm,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_sort
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    : private detail::select_block_sort_impl<Algorithm>::template type<Key, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Value>
#endif
{
    using base_type = typename detail::select_block_sort_impl<Algorithm>::template type<Key, BlockSizeX, BlockSizeY, BlockSizeZ, ItemsPerThread, Value>;
public:








    using storage_type = typename base_type::storage_type;












    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key& thread_key,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, compare_function);
    }

    template <class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_keys, compare_function);
    }










































    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, storage, compare_function);
    }

    template <class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE 
    void sort(Key (&thread_keys)[ItemsPerThread],
              storage_type&  storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_keys, storage, compare_function);
    }













    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, compare_function);
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_keys, thread_values, compare_function);
    }











































    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, storage, compare_function);
    }

    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_keys, thread_values, storage, compare_function);
    }















    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key& thread_key,
              storage_type& storage,
              const unsigned int size,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, storage, size, compare_function);
    }
















    template<class BinaryFunction = ::rocprim::less<Key>>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              const unsigned int size,
              BinaryFunction compare_function = BinaryFunction())
    {
        base_type::sort(thread_key, thread_value, storage, size, compare_function);
    }
};

END_ROCPRIM_NAMESPACE




#endif
