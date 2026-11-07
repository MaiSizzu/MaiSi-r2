



















#ifndef ROCPRIM_TYPES_FUTURE_VALUE_HPP_
#define ROCPRIM_TYPES_FUTURE_VALUE_HPP_

#include "../config.hpp"




BEGIN_ROCPRIM_NAMESPACE


template <typename T, typename Iter = T*>
class future_value
{
public:
    using value_type    = T;
    using iterator_type = Iter;

    explicit ROCPRIM_HOST_DEVICE future_value(const Iter iter)
        : iter_ {iter}
    {
    }

    ROCPRIM_HOST_DEVICE operator T()
    {
        return *iter_;
    }

    ROCPRIM_HOST_DEVICE operator T() const
    {
        return *iter_;
    }
private:
    Iter iter_;
};

namespace detail
{


    template <typename T>
    ROCPRIM_HOST_DEVICE T get_input_value(const T value)
    {
        return value;
    }

    template <typename T, typename Iter>
    ROCPRIM_HOST_DEVICE T get_input_value(::rocprim::future_value<T, Iter> future) {
        return future;
    }

    template <class T>
    struct input_value_traits {
        using value_type = T;
    };

    template <class T, typename Iter> 
    struct input_value_traits<::rocprim::future_value<T, Iter>>
    {
        using value_type    = T;
        using iterator_type = Iter;
    };

    template <typename T>
    using input_type_t = typename input_value_traits<T>::value_type;

    template <typename T>
    using input_iterator_t = typename input_value_traits<T>::iterator_type;
}

END_ROCPRIM_NAMESPACE




#endif
