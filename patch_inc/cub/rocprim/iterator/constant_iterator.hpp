



















#ifndef ROCPRIM_ITERATOR_CONSTANT_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_CONSTANT_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"




BEGIN_ROCPRIM_NAMESPACE












template<
    class ValueType,
    class Difference = std::ptrdiff_t
>
class constant_iterator
{
public:

    using value_type = typename std::remove_const<ValueType>::type;



    using reference = value_type;


    using pointer = const value_type*;

    using difference_type = Difference;

    using iterator_category = std::random_access_iterator_tag;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = constant_iterator;
#endif





    ROCPRIM_HOST_DEVICE inline
    explicit constant_iterator(const value_type value, const size_t index = 0)
        : value_(value), index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~constant_iterator() = default;


    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        return value_;
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &value_;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator++()
    {
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator++(int)
    {
        constant_iterator old_ci = *this;
        index_++;
        return old_ci;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator--()
    {
        index_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator--(int)
    {
        constant_iterator old_ci = *this;
        index_--;
        return old_ci;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator+(difference_type distance) const
    {
        return constant_iterator(value_, index_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator+=(difference_type distance)
    {
        index_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator-(difference_type distance) const
    {
        return constant_iterator(value_, index_ - distance);
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator-=(difference_type distance)
    {
        index_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(constant_iterator other) const
    {
        return static_cast<difference_type>(index_ - other.index_);
    }





    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type) const
    {
        return value_;
    }


    ROCPRIM_HOST_DEVICE inline
    bool operator==(constant_iterator other) const
    {
        return value_ == other.value_ && index_ == other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(constant_iterator other) const
    {
        return !(*this == other);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(constant_iterator other) const
    {
        return distance_to(other) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(constant_iterator other) const
    {
        return distance_to(other) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(constant_iterator other) const
    {
        return distance_to(other) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(constant_iterator other) const
    {
        return distance_to(other) <= 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const constant_iterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }


private:
    inline
    difference_type distance_to(const constant_iterator& other) const
    {
        return difference_type(other.index_) - difference_type(index_);
    }

    value_type value_;
    size_t index_;
};

template<
    class ValueType,
    class Difference
>
ROCPRIM_HOST_DEVICE inline
constant_iterator<ValueType, Difference>
operator+(typename constant_iterator<ValueType, Difference>::difference_type distance,
          const constant_iterator<ValueType, Difference>& iter)
{
    return iter + distance;
}









template<
    class ValueType,
    class Difference = std::ptrdiff_t
>
ROCPRIM_HOST_DEVICE inline
constant_iterator<ValueType, Difference>
make_constant_iterator(ValueType value, size_t index = 0)
{
    return constant_iterator<ValueType, Difference>(value, index);
}

END_ROCPRIM_NAMESPACE




#endif
