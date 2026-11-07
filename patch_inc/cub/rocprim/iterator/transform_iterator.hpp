



















#ifndef ROCPRIM_ITERATOR_TRANSFORM_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_TRANSFORM_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../detail/match_result_type.hpp"




BEGIN_ROCPRIM_NAMESPACE















template<
    class InputIterator,
    class UnaryFunction,
    class ValueType =
        typename ::rocprim::detail::invoke_result<
            UnaryFunction, typename std::iterator_traits<InputIterator>::value_type
        >::type
>
class transform_iterator
{
public:

    using value_type = ValueType;


    using reference = const value_type&;


    using pointer = const value_type*;

    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;

    using iterator_category = std::random_access_iterator_tag;

    using unary_function = UnaryFunction;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = transform_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline
    ~transform_iterator() = default;






    ROCPRIM_HOST_DEVICE inline
    transform_iterator(InputIterator iterator, UnaryFunction transform)
        : iterator_(iterator), transform_(transform)
    {
    }


    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator++()
    {
        iterator_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator++(int)
    {
        transform_iterator old = *this;
        iterator_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator--()
    {
        iterator_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator--(int)
    {
        transform_iterator old = *this;
        iterator_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        return transform_(*iterator_);
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &(*(*this));
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        transform_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator+(difference_type distance) const
    {
        return transform_iterator(iterator_ + distance, transform_);
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator-(difference_type distance) const
    {
        return transform_iterator(iterator_ - distance, transform_);
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(transform_iterator other) const
    {
        return iterator_ - other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(transform_iterator other) const
    {
        return iterator_ == other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(transform_iterator other) const
    {
        return iterator_ != other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(transform_iterator other) const
    {
        return iterator_ < other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(transform_iterator other) const
    {
        return iterator_ <= other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(transform_iterator other) const
    {
        return iterator_ > other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(transform_iterator other) const
    {
        return iterator_ >= other.iterator_;
    }

    friend std::ostream& operator<<(std::ostream& os, const transform_iterator& )
    {
        return os;
    }


private:
    InputIterator iterator_;
    UnaryFunction transform_;
};

template<
    class InputIterator,
    class UnaryFunction,
    class ValueType
>
ROCPRIM_HOST_DEVICE inline
transform_iterator<InputIterator, UnaryFunction, ValueType>
operator+(typename transform_iterator<InputIterator, UnaryFunction, ValueType>::difference_type distance,
          const transform_iterator<InputIterator, UnaryFunction, ValueType>& iterator)
{
    return iterator + distance;
}











template<
    class InputIterator,
    class UnaryFunction
>
ROCPRIM_HOST_DEVICE inline
transform_iterator<InputIterator, UnaryFunction>
make_transform_iterator(InputIterator iterator, UnaryFunction transform)
{
    return transform_iterator<InputIterator, UnaryFunction>(iterator, transform);
}




END_ROCPRIM_NAMESPACE

#endif
