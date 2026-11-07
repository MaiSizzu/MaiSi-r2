



















#ifndef ROCPRIM_ITERATOR_DISCARD_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_DISCARD_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"




BEGIN_ROCPRIM_NAMESPACE









class discard_iterator
{
public:
    struct discard_value
    {
        ROCPRIM_HOST_DEVICE inline
        discard_value() = default;

        template<class T>
        ROCPRIM_HOST_DEVICE inline
        discard_value(T) {};

        ROCPRIM_HOST_DEVICE inline
        ~discard_value() = default;

        template<class T>
        ROCPRIM_HOST_DEVICE inline
        discard_value& operator=(const T&)
        {
            return *this;
        }
    };


    using value_type = discard_value;

    using reference = discard_value;

    using pointer = discard_value*;

    using difference_type = std::ptrdiff_t;

    using iterator_category = std::random_access_iterator_tag;




    ROCPRIM_HOST_DEVICE inline
    discard_iterator(size_t index = 0)
        : index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~discard_iterator() = default;


    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator++()
    {
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator++(int)
    {
        discard_iterator old = *this;
        index_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator--()
    {
        index_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator--(int)
    {
        discard_iterator old = *this;
        index_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_value operator*() const
    {
        return discard_value();
    }

    ROCPRIM_HOST_DEVICE inline
    discard_value operator[](difference_type distance) const
    {
        discard_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator+(difference_type distance) const
    {
        auto i = static_cast<size_t>(static_cast<difference_type>(index_) + distance);
        return discard_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator+=(difference_type distance)
    {
        index_ = static_cast<size_t>(static_cast<difference_type>(index_) + distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator-(difference_type distance) const
    {
        auto i = static_cast<size_t>(static_cast<difference_type>(index_) - distance);
        return discard_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator-=(difference_type distance)
    {
        index_ = static_cast<size_t>(static_cast<difference_type>(index_) - distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(discard_iterator other) const
    {
        return index_ - other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(discard_iterator other) const
    {
        return index_ == other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(discard_iterator other) const
    {
        return index_ != other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(discard_iterator other) const
    {
        return index_ < other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(discard_iterator other) const
    {
        return index_ <= other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(discard_iterator other) const
    {
        return index_ > other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(discard_iterator other) const
    {
        return index_ >= other.index_;
    }

    friend std::ostream& operator<<(std::ostream& os, const discard_iterator& )
    {
        return os;
    }


private:
    mutable size_t index_;
};

ROCPRIM_HOST_DEVICE inline
discard_iterator
operator+(typename discard_iterator::difference_type distance,
          const discard_iterator& iterator)
{
    return iterator + distance;
}






ROCPRIM_HOST_DEVICE inline
discard_iterator
make_discard_iterator(size_t index = 0)
{
    return discard_iterator(index);
}




END_ROCPRIM_NAMESPACE

#endif
