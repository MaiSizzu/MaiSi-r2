



















#ifndef ROCPRIM_BLOCK_BLOCK_LOAD_HPP_
#define ROCPRIM_BLOCK_BLOCK_LOAD_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_load_func.hpp"
#include "block_exchange.hpp"




BEGIN_ROCPRIM_NAMESPACE



enum class block_load_method
{




    block_load_direct,


    block_load_striped,














    block_load_vectorize,








    block_load_transpose,










    block_load_warp_transpose,


    default_method = block_load_direct
};



































template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    block_load_method Method = block_load_method::block_load_direct,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_load
{
private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:








    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_;
    #endif













    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items);
    }














    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid);
    }

















    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid,
                                  out_of_bounds);
    }































    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items);
    }
































    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid);
    }



































    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid, out_of_bounds);
    }
};




#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
    >
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_striped, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_;
    #endif

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
    }

    template<
        class InputIterator,
        class Default
        >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid,
                                             out_of_bounds);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
    }

    template<
        class InputIterator,
        class Default
        >
    ROCPRIM_DEVICE inline
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid,
                                             out_of_bounds);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_vectorize, BlockSizeY, BlockSizeZ>
{
private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_;
    #endif

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(T* block_input,
              T (&_items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked_vectorized(flat_id, block_input, _items);
    }

    template<class InputIterator, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              U (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_blocked(flat_id, block_input, items, valid,
                                  out_of_bounds);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(T* block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        (void) storage;
        load(block_input, items);
    }

    template<class InputIterator, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              U (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        (void) storage;
        load(block_input, items, valid, out_of_bounds);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_transpose, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

private:
    using block_exchange_type = block_exchange<T, BlockSize, ItemsPerThread>;

public:
    using storage_type = typename block_exchange_type::storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid,
                                             out_of_bounds);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_striped<BlockSize>(flat_id, block_input, items, valid,
                                             out_of_bounds);
        block_exchange_type().striped_to_blocked(items, items, storage);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_load<T, BlockSizeX, ItemsPerThread, block_load_method::block_load_warp_transpose, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
private:
    using block_exchange_type = block_exchange<T, BlockSizeX, ItemsPerThread, BlockSizeY, BlockSizeZ>;

public:
    static_assert(BlockSize % ::rocprim::device_warp_size() == 0,
                 "BlockSize must be a multiple of hardware warpsize");

    using storage_type = typename block_exchange_type::storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread])
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);

    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid,
                                       out_of_bounds);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator block_input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_load_direct_warp_striped(flat_id, block_input, items, valid,
                                       out_of_bounds);
        block_exchange_type().warp_striped_to_blocked(items, items, storage);
    }
};

#endif

END_ROCPRIM_NAMESPACE

#endif
