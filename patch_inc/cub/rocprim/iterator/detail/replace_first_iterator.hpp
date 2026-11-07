



















#ifndef ROCPRIM_ITERATOR_REPLACE_FIRST_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_REPLACE_FIRST_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{















template<class InputIterator>
class replace_first_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;
    static_assert(
        std::is_same<input_category, std::random_access_iterator_tag>::value,
        "InputIterator must be a random-access iterator"
    );

public:
    using value_type = typename std::iterator_traits<InputIterator>::value_type;
    using reference = value_type;
    using pointer = const value_type*;
    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline
    ~replace_first_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator(InputIterator iterator, value_type value, size_t index = 0)
        : iterator_(iterator), value_(value), index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator& operator++()
    {
        iterator_++;
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator operator++(int)
    {
        replace_first_iterator old = *this;
        iterator_++;
        index_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        if(index_ == 0)
        {
            return value_;
        }
        return *iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        replace_first_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator operator+(difference_type distance) const
    {
        return replace_first_iterator(iterator_ + distance, value_, index_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        index_ += distance;
        return *this;
    }

private:
    InputIterator iterator_;
    value_type value_;
    size_t index_;
};

}

END_ROCPRIM_NAMESPACE

#endif
