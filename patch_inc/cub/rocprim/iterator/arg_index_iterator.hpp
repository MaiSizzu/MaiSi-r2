



















#ifndef ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../types/key_value_pair.hpp"




BEGIN_ROCPRIM_NAMESPACE















template<
    class InputIterator,
    class Difference = std::ptrdiff_t,
    class InputValueType = typename std::iterator_traits<InputIterator>::value_type
>
class arg_index_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;

public:

    using value_type = ::rocprim::key_value_pair<Difference, InputValueType>;


    using reference = const value_type&;


    using pointer = const value_type*;

    using difference_type = Difference;

    using iterator_category = std::random_access_iterator_tag;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = arg_index_iterator;
#endif

    static_assert(
        std::is_same<input_category, iterator_category>::value,
        "InputIterator must be a random-access iterator"
    );

    ROCPRIM_HOST_DEVICE inline
    ~arg_index_iterator() = default;





    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator(InputIterator iterator, difference_type offset = 0)
        : iterator_(iterator), offset_(offset)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator++()
    {
        iterator_++;
        offset_++;
        return *this;
    }


    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator++(int)
    {
        arg_index_iterator old_ai = *this;
        iterator_++;
        offset_++;
        return old_ai;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        value_type ret(offset_, *iterator_);
        return ret;
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &(*(*this));
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator+(difference_type distance) const
    {
        return arg_index_iterator(iterator_ + distance, offset_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        offset_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator-(difference_type distance) const
    {
        return arg_index_iterator(iterator_ - distance, offset_ - distance);
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        offset_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(arg_index_iterator other) const
    {
        return iterator_ - other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        arg_index_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(arg_index_iterator other) const
    {
        return (iterator_ == other.iterator_) && (offset_ == other.offset_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(arg_index_iterator other) const
    {
        return (iterator_ != other.iterator_) || (offset_ != other.offset_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) <= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    void normalize()
    {
        offset_ = 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const arg_index_iterator& )
    {
        return os;
    }


private:
    InputIterator iterator_;
    difference_type offset_;
};

template<
    class InputIterator,
    class Difference,
    class InputValueType
>
ROCPRIM_HOST_DEVICE inline
arg_index_iterator<InputIterator, Difference, InputValueType>
operator+(typename arg_index_iterator<InputIterator, Difference, InputValueType>::difference_type distance,
          const arg_index_iterator<InputIterator, Difference, InputValueType>& iterator)
{
    return iterator + distance;
}















template<
    class InputIterator,
    class Difference = std::ptrdiff_t,
    class InputValueType = typename std::iterator_traits<InputIterator>::value_type
>
ROCPRIM_HOST_DEVICE inline
arg_index_iterator<InputIterator, Difference, InputValueType>
make_arg_index_iterator(InputIterator iterator, Difference offset = 0)
{
    return arg_index_iterator<InputIterator, Difference, InputValueType>(iterator, offset);
}

END_ROCPRIM_NAMESPACE




#endif
