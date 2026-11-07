



















#ifndef ROCPRIM_TYPES_INTEGER_SEQUENCE_HPP_
#define ROCPRIM_TYPES_INTEGER_SEQUENCE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
#if defined(__cpp_lib_integer_sequence) && !defined(DOXYGEN_SHOULD_SKIP_THIS)

using std::integer_sequence;
using std::index_sequence;
using std::make_integer_sequence;
using std::make_index_sequence;
using std::index_sequence_for;
#else




template<class T, T... Ints>
class integer_sequence
{
    using value_type = T;

    static inline constexpr size_t size() noexcept
    {
        return sizeof...(Ints);
    }
};

template<size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;


namespace detail
{

template<class T, class IntegerSequence>
struct integer_sequence_cat;

template<class T, T... Indices>
struct integer_sequence_cat<T, ::rocprim::integer_sequence<T, Indices...>>
{
    using type = typename ::rocprim::integer_sequence<T, Indices..., sizeof...(Indices)>;
};

template<class T, size_t Count>
struct make_integer_sequence_impl :
    integer_sequence_cat<T, typename make_integer_sequence_impl<T, Count - 1>::type>
{
};

template<class T>
struct make_integer_sequence_impl<T, 0>
{
    using type = ::rocprim::integer_sequence<T>;
};

}

template<class T, T N>
using make_integer_sequence = typename detail::make_integer_sequence_impl<T, N>::type;

template<size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

template<class... T>
using index_sequence_for = make_index_sequence<sizeof...(T)>;
#endif

END_ROCPRIM_NAMESPACE

#endif
