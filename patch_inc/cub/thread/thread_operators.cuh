

#ifndef HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_
#define HIBCUB_ROCPRIM_THREAD_THREAD_OPERATORS_HPP_

#include "../config.hpp"

#include "../util_type.cuh"

BEGIN_HIPCUB_NAMESPACE

struct Equality
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

struct Inequality
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template <class EqualityOp>
struct InequalityWrapper
{
    EqualityOp op;

    HIPCUB_HOST_DEVICE inline
    InequalityWrapper(EqualityOp op) : op(op) {}

    template<class T>
    HIPCUB_HOST_DEVICE inline
    bool operator()(const T &a, const T &b)
    {
        return !op(a, b);
    }
};

struct Sum
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

struct Difference
{
    template <class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a - b;
    }
};

struct Division
{
    template <class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a / b;
    }
};

struct Max
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? b : a;
    }
};

struct Min
{
    template<class T>
    HIPCUB_HOST_DEVICE inline
    constexpr T operator()(const T &a, const T &b) const
    {
        return a < b ? a : b;
    }
};

struct ArgMax
{
    template<
        class Key,
        class Value
    >
    HIPCUB_HOST_DEVICE inline
    constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a,
               const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

struct ArgMin
{
    template<
        class Key,
        class Value
    >
    HIPCUB_HOST_DEVICE inline
    constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a,
               const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

template <typename B>
struct CastOp
{
    template <typename A>
    HIPCUB_HOST_DEVICE inline
    B operator()(const A &a) const
    {
        return (B)a;
    }
};

template <typename ScanOp>
class SwizzleScanOp
{
private:
    ScanOp scan_op;

public:
    HIPCUB_HOST_DEVICE inline
    SwizzleScanOp(ScanOp scan_op) : scan_op(scan_op)
    {
    }

    template <typename T>
    HIPCUB_HOST_DEVICE inline
    T operator()(const T &a, const T &b)
    {
      T _a(a);
      T _b(b);

      return scan_op(_b, _a);
    }
};

template <typename ReductionOpT>
struct ReduceBySegmentOp
{
    ReductionOpT op;

    HIPCUB_HOST_DEVICE inline
    ReduceBySegmentOp()
    {
    }

    HIPCUB_HOST_DEVICE inline
    ReduceBySegmentOp(ReductionOpT op) : op(op)
    {
    }

    template <typename KeyValuePairT>
    HIPCUB_HOST_DEVICE inline
    KeyValuePairT operator()(
        const KeyValuePairT &first,
        const KeyValuePairT &second)
    {
        KeyValuePairT retval;
        retval.key = first.key + second.key;
        retval.value = (second.key) ?
                second.value :
                op(first.value, second.value);
        return retval;
    }
};

template <typename ReductionOpT>
struct ReduceByKeyOp
{
    ReductionOpT op;

    HIPCUB_HOST_DEVICE inline
    ReduceByKeyOp()
    {
    }

    HIPCUB_HOST_DEVICE inline
    ReduceByKeyOp(ReductionOpT op) : op(op)
    {
    }

    template <typename KeyValuePairT>
    HIPCUB_HOST_DEVICE inline
    KeyValuePairT operator()(
        const KeyValuePairT &first,
        const KeyValuePairT &second)
    {
        KeyValuePairT retval = second;

        if (first.key == second.key)
        {
            retval.value = op(first.value, retval.value);
        }
        return retval;
    }
};

template <typename BinaryOpT>
struct BinaryFlip
{
    BinaryOpT binary_op;

    HIPCUB_HOST_DEVICE
    explicit BinaryFlip(BinaryOpT binary_op) : binary_op(binary_op)
    {
    }

    template <typename T, typename U>
    HIPCUB_DEVICE auto
    operator()(T &&t, U &&u) -> decltype(binary_op(std::forward<U>(u),
                                                    std::forward<T>(t)))
    {
        return binary_op(std::forward<U>(u), std::forward<T>(t));
    }
};

template <typename BinaryOpT>
HIPCUB_HOST_DEVICE
BinaryFlip<BinaryOpT> MakeBinaryFlip(BinaryOpT binary_op)
{
    return BinaryFlip<BinaryOpT>(binary_op);
}

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
    HIPCUB_HOST_DEVICE inline
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

END_HIPCUB_NAMESPACE

#endif
