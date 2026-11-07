



















#ifndef ROCPRIM_ITERATOR_REVERSE_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_REVERSE_ITERATOR_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "../config.hpp"




BEGIN_ROCPRIM_NAMESPACE











template<class SourceIterator>
class reverse_iterator
{
public:
    static_assert(
        std::is_base_of<std::random_access_iterator_tag,
                        typename std::iterator_traits<SourceIterator>::iterator_category>::value,
        "SourceIterator must be a random access iterator");


    using value_type = typename std::iterator_traits<SourceIterator>::value_type;

    using reference = typename std::iterator_traits<SourceIterator>::reference;

    using pointer = typename std::iterator_traits<SourceIterator>::pointer;

    using difference_type = typename std::iterator_traits<SourceIterator>::difference_type;

    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE
    reverse_iterator(SourceIterator source_iterator) : source_iterator_(source_iterator) {}


    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator++()
    {
        --source_iterator_;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator++(int)
    {
        reverse_iterator old = *this;
        --source_iterator_;
        return old;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator--()
    {
        ++source_iterator_;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator--(int)
    {
        reverse_iterator old = *this;
        ++source_iterator_;
        return old;
    }

    ROCPRIM_HOST_DEVICE
    reference operator*()
    {
        return *(source_iterator_ - static_cast<difference_type>(1));
    }

    ROCPRIM_HOST_DEVICE
    reference operator[](difference_type distance)
    {
        reverse_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator+(difference_type distance) const
    {
        return reverse_iterator(source_iterator_ - distance);
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator+=(difference_type distance)
    {
        source_iterator_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator-(difference_type distance) const
    {
        return reverse_iterator(source_iterator_ + distance);
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator-=(difference_type distance)
    {
        source_iterator_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    difference_type operator-(reverse_iterator other) const
    {
        return other.source_iterator_ - source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator==(reverse_iterator other) const
    {
        return source_iterator_ == other.source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator!=(reverse_iterator other) const
    {
        return source_iterator_ != other.source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator<(reverse_iterator other) const
    {
        return other.source_iterator_ < source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator<=(reverse_iterator other) const
    {
        return other.source_iterator_ <= source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator>(reverse_iterator other) const
    {
        return other.source_iterator_ > source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator>=(reverse_iterator other) const
    {
        return other.source_iterator_ >= source_iterator_;
    }


private:
    SourceIterator source_iterator_;
};

template<class SourceIterator>
ROCPRIM_HOST_DEVICE reverse_iterator<SourceIterator>
                    operator+(typename reverse_iterator<SourceIterator>::difference_type distance,
              const reverse_iterator<SourceIterator>&                    iterator)
{
    return iterator + distance;
}







template<class SourceIterator>
ROCPRIM_HOST_DEVICE reverse_iterator<SourceIterator>
                    make_reverse_iterator(SourceIterator source_iterator)
{
    return reverse_iterator<SourceIterator>(source_iterator);
}




END_ROCPRIM_NAMESPACE

#endif
