



















#ifndef ROCPRIM_WARP_WARP_LOAD_HPP_
#define ROCPRIM_WARP_WARP_LOAD_HPP_

#include "../config.hpp"
#include "../intrinsics.hpp"
#include "../detail/various.hpp"

#include "warp_exchange.hpp"
#include "../block/block_load_func.hpp"




BEGIN_ROCPRIM_NAMESPACE



enum class warp_load_method
{




    warp_load_direct,


    warp_load_striped,














    warp_load_vectorize,








    warp_load_transpose,


    default_method = warp_load_direct
};








































template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize = ::rocprim::device_warp_size(),
    warp_load_method Method = warp_load_method::warp_load_direct
>
class warp_load
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














    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked(flat_id, input, items);
    }















    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked(flat_id, input, items, valid);
    }
















    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked(flat_id, input, items, valid,
                                  out_of_bounds);
    }
};




#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_load<T, ItemsPerThread, WarpSize, warp_load_method::warp_load_striped>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_warp_striped<WarpSize>(flat_id, input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_warp_striped<WarpSize>(flat_id, input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_warp_striped<WarpSize>(flat_id, input, items, valid,
                                                 out_of_bounds);
    }
};

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_load<T, ItemsPerThread, WarpSize, warp_load_method::warp_load_vectorize>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(T* input,
              T (&items)[ItemsPerThread],
              storage_type& )
    {
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked_vectorized(flat_id, input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked(flat_id, input, items);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked(flat_id, input, items, valid);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& )
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_blocked(flat_id, input, items, valid,
                                  out_of_bounds);
    }
};

template<
    class T,
    unsigned int ItemsPerThread,
    unsigned int WarpSize
>
class warp_load<T, ItemsPerThread, WarpSize, warp_load_method::warp_load_transpose>
{
    static_assert(::rocprim::detail::is_power_of_two(WarpSize),
                  "Logical warp size must be a power of two.");
    static_assert(WarpSize <= ::rocprim::device_warp_size(),
                  "Logical warp size cannot be larger than physical warp size.");

private:
    using exchange_type = ::rocprim::warp_exchange<T, ItemsPerThread, WarpSize>;

public:
    using storage_type = typename exchange_type::storage_type;

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_warp_striped<WarpSize>(flat_id, input, items);
        exchange_type().striped_to_blocked(items, items, storage);
    }

    template<class InputIterator>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_warp_striped<WarpSize>(flat_id, input, items, valid);
        exchange_type().striped_to_blocked(items, items, storage);
    }

    template<
        class InputIterator,
        class Default
    >
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void load(InputIterator input,
              T (&items)[ItemsPerThread],
              unsigned int valid,
              Default out_of_bounds,
              storage_type& storage)
    {
        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        static_assert(std::is_convertible<value_type, T>::value,
                      "The type T must be such that an object of type InputIterator "
                      "can be dereferenced and then implicitly converted to T.");
        const unsigned int flat_id = ::rocprim::detail::logical_lane_id<WarpSize>();
        block_load_direct_warp_striped<WarpSize>(flat_id, input, items, valid,
                                                 out_of_bounds);
        exchange_type().striped_to_blocked(items, items, storage);
    }
};

#endif

END_ROCPRIM_NAMESPACE

#endif
