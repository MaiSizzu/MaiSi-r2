



















#ifndef ROCPRIM_WARP_WARP_SORT_HPP_
#define ROCPRIM_WARP_WARP_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"

#include "detail/warp_sort_shuffle.hpp"




BEGIN_ROCPRIM_NAMESPACE
























































template<
    class Key,
    unsigned int WarpSize = device_warp_size(),
    class Value = empty_type
>
class warp_sort : detail::warp_sort_shuffle<Key, WarpSize, Value>
{
    typedef typename detail::warp_sort_shuffle<Key, WarpSize, Value> base_type;


    static_assert(WarpSize <= ROCPRIM_MAX_WARP_SIZE, "WarpSize can't be greater than hardware warp size.");

public:








    typedef typename base_type::storage_type storage_type;











    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& thread_key,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(thread_key, compare_function);
    }



    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& ,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }











    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(thread_keys, compare_function);
    }



    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) thread_keys;
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }





























    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& thread_key,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(
            thread_key, storage, compare_function
        );
    }



    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& ,
              storage_type& ,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }






























    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(
            thread_keys, storage, compare_function
        );
    }



    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              storage_type& ,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) thread_keys;
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }












    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& thread_key,
              Value& thread_value,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(
            thread_key, thread_value, compare_function
        );
    }



    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& ,
              Value& ,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }












    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(
            thread_keys, thread_values, compare_function
        );
    }



    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) thread_keys;
        (void) thread_values;
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }






























    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& thread_key,
              Value& thread_value,
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(
            thread_key, thread_value, storage, compare_function
        );
    }



    template<class BinaryFunction = ::rocprim::less<Key>, unsigned int FunctionWarpSize = WarpSize>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key& ,
              Value& ,
              storage_type& ,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }































    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type& storage,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize <= __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        base_type::sort(
            thread_keys, thread_values, storage, compare_function
        );
    }



    template<
        unsigned int ItemsPerThread,
        class BinaryFunction = ::rocprim::less<Key>,
        unsigned int FunctionWarpSize = WarpSize
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    auto sort(Key (&thread_keys)[ItemsPerThread],
              Value (&thread_values)[ItemsPerThread],
              storage_type& ,
              BinaryFunction compare_function = BinaryFunction())
        -> typename std::enable_if<(FunctionWarpSize > __AMDGCN_WAVEFRONT_SIZE), void>::type
    {
        (void) thread_keys;
        (void) thread_values;
        (void) compare_function;
        ROCPRIM_PRINT_ERROR_ONCE("Specified warp size exceeds current hardware supported warp size. Aborting warp sort.");
        return;
    }
};

END_ROCPRIM_NAMESPACE




#endif
