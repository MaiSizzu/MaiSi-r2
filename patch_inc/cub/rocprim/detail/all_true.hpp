



















#ifndef ROCPRIM_DETAIL_ALL_TRUE_HPP_
#define ROCPRIM_DETAIL_ALL_TRUE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{



template<bool... values>
struct all_true : std::true_type
{
};

template<bool... values>
struct all_true<true, values...> : all_true<values...>
{
};

template<bool... values>
struct all_true<false, values...> : std::false_type
{
};

}
END_ROCPRIM_NAMESPACE

#endif
