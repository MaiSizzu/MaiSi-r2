



















#ifndef ROCPRIM_WARP_WARP_STORE_HPP_
#define ROCPRIM_WARP_WARP_STORE_HPP_

#include "../config.hpp"
#include "../intrinsics.hpp"
#include "../detail/various.hpp"

#include "warp_exchange.hpp"
#include "../block/block_store_func.hpp"




BEGIN_ROCPRIM_NAMESPACE



enum class warp_store_method
{





    warp_store_direct,



    warp_store_striped,














    warp_store_vectorize,








    warp_store_transpose,


    default_method = warp_store_direct
};








































template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize = ::rocprim::device_warp_size(),
    warp_store_method Method = warp_store_method::warp_store_direct
>
class warp_store
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

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
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& )
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items);
    }




















    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& )
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items, valid);
    }
};




#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_store<T, ItemsPerThread, WarpSize, warp_store_method::warp_store_striped>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& )
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& )
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items, valid);
    }
};

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_store<T, ItemsPerThread, WarpSize, warp_store_method::warp_store_vectorize>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(T* output,
               T (&items)[ItemsPerThread],
               storage_type& )
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked_vectorized(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& )
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& )
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_blocked(flat_id, output, items, valid);
    }
};

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_store<T, ItemsPerThread, WarpSize, warp_store_method::warp_store_transpose>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

private:
    using exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, WarpSize>;

public:
    using storage_type = typename exchange_type::storage_type;

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               storage_type& storage)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        exchange_type().blocked_to_striped(items, items, storage);
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items);
    }

    template<class OutputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void store(OutputIterator output,
               T (&items)[ItemsPerThread],
               unsigned int valid,
               storage_type& storage)
    {
        using value_type = typename std::iterator_traits<OutputIterator>::value_type;
        static_assert(std::is_convertible<T, value_type>::value,
                      "The type T must be such that an object of type OutputIterator "
                      "can be dereferenced and then implicitly assigned from T.");
        exchange_type().blocked_to_striped(items, items, storage);
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_store_direct_warp_striped<WarpSize>(flat_id, output, items, valid);
    }
};

#endif

END_ROCPRIM_NAMESPACE

#endif
