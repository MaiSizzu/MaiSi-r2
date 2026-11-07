



















#ifndef ROCPRIM_BLOCK_BLOCK_STORE_HPP_
#define ROCPRIM_BLOCK_BLOCK_STORE_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_store_func.hpp"
#include "block_exchange.hpp"




BEGIN_ROCPRIM_NAMESPACE



enum class block_store_method
{





    block_store_direct,



    block_store_striped,














    block_store_vectorize,








    block_store_transpose,










    block_store_warp_transpose,


    default_method = block_store_direct
};


































template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    block_store_method Method = block_store_method::block_store_direct,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_store
{
private:
    using storage_type_ = typename ::rocprim::detail::empty_storage_type;

public:








    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = typename ::rocprim::detail::empty_storage_type;
    #else
    using storage_type = storage_type_;
    #endif













    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_blocked(flat_id, block_output, items);
    }














    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_blocked(flat_id, block_output, items, valid);
    }































    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items);
    }

































    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items, valid);
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
class block_store<T, BlockSizeX, ItemsPerThread, block_store_method::block_store_striped, BlockSizeY, BlockSizeZ>
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

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE inline
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        (void) storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_store<T, BlockSizeX, ItemsPerThread, block_store_method::block_store_vectorize, BlockSizeY, BlockSizeZ>
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
    void store(T* block_output,
               T (&_items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_blocked_vectorized(flat_id, block_output, _items);
    }

    template<class OutputIterator, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               U (&items)[ItemsPerThread])
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_blocked(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_store_direct_blocked(flat_id, block_output, items, valid);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(T* block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items);
    }

    template<class OutputIterator, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               U (&items)[ItemsPerThread],
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        (void) storage;
        store(block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_store<T, BlockSizeX, ItemsPerThread, block_store_method::block_store_transpose, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
private:
    using block_exchange_type = block_exchange<T, BlockSize, ItemsPerThread>;

public:
    using storage_type = typename block_exchange_type::storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_striped(items, items, storage);
        block_store_direct_striped<BlockSize>(flat_id, block_output, items, valid);
    }
};

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ
>
class block_store<T, BlockSizeX, ItemsPerThread, block_store_method::block_store_warp_transpose, BlockSizeY, BlockSizeZ>
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
private:
    using block_exchange_type = block_exchange<T, BlockSize, ItemsPerThread>;

public:
    static_assert(BlockSize % ::rocprim::device_warp_size() == 0,
                 "BlockSize must be a multiple of hardware warpsize");

    using storage_type = typename block_exchange_type::storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid)
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items, valid);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator block_output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        block_exchange_type().blocked_to_warp_striped(items, items, storage);
        block_store_direct_warp_striped(flat_id, block_output, items, valid);
    }
};

#endif

END_ROCPRIM_NAMESPACE

#endif
