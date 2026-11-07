



















#ifndef ROCPRIM_DETAIL_BINARY_OP_WRAPPERS_HPP_
#define ROCPRIM_DETAIL_BINARY_OP_WRAPPERS_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../intrinsics.hpp"
#include "../types.hpp"
#include "../functional.hpp"

#include "../detail/various.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class BinaryFunction,
    class ResultType = typename BinaryFunction::result_type,
    class InputType = typename BinaryFunction::input_type
>
struct reverse_binary_op_wrapper
{
    using result_type = ResultType;
    using input_type  = InputType;

    ROCPRIM_HOST_DEVICE inline
    reverse_binary_op_wrapper() = default;

    ROCPRIM_HOST_DEVICE inline
    reverse_binary_op_wrapper(BinaryFunction binary_op)
        : binary_op_(binary_op)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~reverse_binary_op_wrapper() = default;

    ROCPRIM_HOST_DEVICE inline
    result_type operator()(const input_type& t1, const input_type& t2)
    {
        return binary_op_(t2, t1);
    }

private:
    BinaryFunction binary_op_;
};


template<class V, class F, class BinaryFunction>
struct headflag_scan_op_wrapper
{
    static_assert(std::is_convertible<F, bool>::value, "F must be convertible to bool");

    using result_type = rocprim::tuple<V, F>;
    using input_type  = result_type;

    ROCPRIM_HOST_DEVICE inline
    headflag_scan_op_wrapper() = default;

    ROCPRIM_HOST_DEVICE inline
    headflag_scan_op_wrapper(BinaryFunction scan_op)
        : scan_op_(scan_op)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~headflag_scan_op_wrapper() = default;

    ROCPRIM_HOST_DEVICE inline
    result_type operator()(const input_type& t1, const input_type& t2)
    {
        return rocprim::make_tuple(!rocprim::get<1>(t2)
                                       ? scan_op_(rocprim::get<0>(t1), rocprim::get<0>(t2))
                                       : rocprim::get<0>(t2),
                                   F {rocprim::get<1>(t2) || rocprim::get<1>(t1)});
    }

private:
    BinaryFunction scan_op_;
};


template<class EqualityOp>
struct inequality_wrapper
{
    using equality_op_type = EqualityOp;

    ROCPRIM_HOST_DEVICE inline
    inequality_wrapper() = default;

    ROCPRIM_HOST_DEVICE inline
    inequality_wrapper(equality_op_type equality_op)
        : equality_op(equality_op)
    {}

    template<class T, class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    bool operator()(const T &a, const U &b)
    {
        return !equality_op(a, b);
    }

    equality_op_type equality_op;
};

}

END_ROCPRIM_NAMESPACE

#endif
