

#ifndef HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_ARG_INDEX_INPUT_ITERATOR_HPP_

#include <iterator>
#include <iostream>

#include "../config.hpp"

#include <cub/rocprim/iterator/arg_index_iterator.hpp>

#if (THRUST_VERSION >= 100700)

    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif

BEGIN_HIPCUB_NAMESPACE

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template<
    typename InputIterator,
    typename Difference = std::ptrdiff_t,
    typename Value = typename std::iterator_traits<InputIterator>::value_type
>
using ArgIndexInputIterator = ::rocprim::arg_index_iterator<InputIterator, Difference, Value>;

#endif

END_HIPCUB_NAMESPACE

#endif
