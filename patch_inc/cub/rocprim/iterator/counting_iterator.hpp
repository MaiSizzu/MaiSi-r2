



















#ifndef ROCPRIM_ITERATOR_COUNTING_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_COUNTING_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../type_traits.hpp"




BEGIN_ROCPRIM_NAMESPACE











template<
    class Incrementable,
    class Difference = std::ptrdiff_t
>
class counting_iterator
{
public:

    using value_type = typename std::remove_const<Incrementable>::type;



    using reference = value_type;


    using pointer = const value_type*;

    using difference_type = Difference;

    using iterator_category = std::random_access_iterator_tag;

    static_assert(std::is_integral<value_type>::value, "Incrementable must be integral type");

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = counting_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline
    counting_iterator() = default;



    ROCPRIM_HOST_DEVICE inline
    ~counting_iterator() = default;




    ROCPRIM_HOST_DEVICE inline
    explicit counting_iterator(const value_type value) : value_(value)
    {
    }


    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator++()
    {
        value_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator operator++(int)
    {
        counting_iterator old_ci = *this;
        value_++;
        return old_ci;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator--()
    {
        value_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator operator--(int)
    {
        counting_iterator old_ci = *this;
        value_--;
        return old_ci;
    }

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
    counting_iterator operator+(difference_type distance) const
    {
        return counting_iterator(value_ + static_cast<value_type>(distance));
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator+=(difference_type distance)
    {
        value_ += static_cast<value_type>(distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator operator-(difference_type distance) const
    {
        return counting_iterator(value_ - static_cast<value_type>(distance));
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator-=(difference_type distance)
    {
        value_ -= static_cast<value_type>(distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(counting_iterator other) const
    {
        return static_cast<difference_type>(value_ - other.value_);
    }




    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        return value_ + static_cast<value_type>(distance);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(counting_iterator other) const
    {
        return this->equal_value(value_, other.value_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(counting_iterator other) const
    {
        return !(*this == other);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(counting_iterator other) const
    {
        return distance_to(other) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(counting_iterator other) const
    {
        return distance_to(other) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(counting_iterator other) const
    {
        return distance_to(other) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(counting_iterator other) const
    {
        return distance_to(other) <= 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const counting_iterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }


private:
    template<class T>
    inline
    bool equal_value(const T& x, const T& y) const
    {
        return (x == y);
    }

    inline
    difference_type distance_to(const counting_iterator& other) const
    {
        return difference_type(other.value_) - difference_type(value_);
    }

    value_type value_;
};

template<
    class Incrementable,
    class Difference
>
ROCPRIM_HOST_DEVICE inline
counting_iterator<Incrementable, Difference>
operator+(typename counting_iterator<Incrementable, Difference>::difference_type distance,
          const counting_iterator<Incrementable, Difference>& iter)
{
    return iter + distance;
}








template<
    class Incrementable,
    class Difference = std::ptrdiff_t
>
ROCPRIM_HOST_DEVICE inline
counting_iterator<Incrementable, Difference>
make_counting_iterator(Incrementable value)
{
    return counting_iterator<Incrementable, Difference>(value);
}

END_ROCPRIM_NAMESPACE




#endif
