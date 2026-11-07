

#ifndef HIPCUB_ROCPRIM_ITERATOR_TRANSFORM_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_TRANSFORM_INPUT_ITERATOR_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"

#include <cub/rocprim/iterator/transform_iterator.hpp>

#if (THRUST_VERSION >= 100700)

    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif


BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    typename ValueType,
    typename ConversionOp,
    typename InputIteratorT,
    typename OffsetT = std::ptrdiff_t
>
using TransformInputIterator = ::rocprim::transform_iterator<InputIteratorT, ConversionOp, ValueType>;

#endif

END_HIPCUB_NAMESPACE

#endif
