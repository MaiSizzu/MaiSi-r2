

#ifndef ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
#define ROCPRIM_THREAD_THREAD_OPERATORS_HPP_

#include "../config.hpp"
#include "../types.hpp"


BEGIN_ROCPRIM_NAMESPACE

struct equality
{
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

struct inequality
{
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template <class EqualityOp>
struct inequality_wrapper
{
    EqualityOp op;

    ROCPRIM_HOST_DEVICE inline
    inequality_wrapper(EqualityOp op) : op(op) {}

    template<class T>
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const T &a, const T &b)
    {
        return !op(a, b);
    }
};

struct sum
{
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

struct max
{
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? b : a;
    }
};

struct min
{
    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? a : b;
    }
};

struct arg_max
{
    template<
        class Key,
        class Value
    >
    ROCPRIM_HOST_DEVICE inline
    constexpr key_value_pair<Key, Value>
    operator()(const key_value_pair<Key, Value>& a,
               const key_value_pair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

struct arg_min
{
    template<
        class Key,
        class Value
    >
    ROCPRIM_HOST_DEVICE inline
    constexpr key_value_pair<Key, Value>
    operator()(const key_value_pair<Key, Value>& a,
               const key_value_pair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

namespace detail
{





















template<
    class InputIteratorT,
    class OutputIteratorT,
    class BinaryFunction
>
struct convert_result_type_wrapper
{
    using input_type = typename std::iterator_traits<InputIteratorT>::value_type;
    using output_type = typename std::iterator_traits<OutputIteratorT>::value_type;
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, input_type, output_type
        >::type;

    convert_result_type_wrapper(BinaryFunction op) : op(op) {}

    template<class T>
    ROCPRIM_HOST_DEVICE inline
    constexpr result_type operator()(const T &a, const T &b) const
    {
        return static_cast<result_type>(op(a, b));
    }

    BinaryFunction op;
};

template<
    class InputIteratorT,
    class OutputIteratorT,
    class BinaryFunction
>
inline
convert_result_type_wrapper<InputIteratorT, OutputIteratorT, BinaryFunction>
convert_result_type(BinaryFunction op)
{
    return convert_result_type_wrapper<InputIteratorT, OutputIteratorT, BinaryFunction>(op);
}

}

END_ROCPRIM_NAMESPACE

#endif
