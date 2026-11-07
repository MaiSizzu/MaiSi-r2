



















#ifndef ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_
#define ROCPRIM_DETAIL_MATCH_RESULT_TYPE_HPP_

#include <type_traits>

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{






template <class T>
struct is_reference_wrapper : std::false_type {};
template <class U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

template<class T>
struct invoke_impl {
    template<class F, class... Args>
    ROCPRIM_HOST_DEVICE
    static auto call(F&& f, Args&&... args)
        -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
};

template<class B, class MT>
struct invoke_impl<MT B::*>
{
    template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<std::is_base_of<B, Td>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto get(T&& t) -> T&&;

    template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<is_reference_wrapper<Td>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto get(T&& t) -> decltype(t.get());

    template<class T, class Td = typename std::decay<T>::type,
        class = typename std::enable_if<!std::is_base_of<B, Td>::value>::type,
        class = typename std::enable_if<!is_reference_wrapper<Td>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto get(T&& t) -> decltype(*std::forward<T>(t));

    template<class T, class... Args, class MT1,
        class = typename std::enable_if<std::is_function<MT1>::value>::type
    >
    ROCPRIM_HOST_DEVICE
    static auto call(MT1 B::*pmf, T&& t, Args&&... args)
        -> decltype((invoke_impl::get(std::forward<T>(t)).*pmf)(std::forward<Args>(args)...));

    template<class T>
    ROCPRIM_HOST_DEVICE
    static auto call(MT B::*pmd, T&& t)
        -> decltype(invoke_impl::get(std::forward<T>(t)).*pmd);
};

template<class F, class... Args, class Fd = typename std::decay<F>::type>
ROCPRIM_HOST_DEVICE
auto INVOKE(F&& f, Args&&... args)
    -> decltype(invoke_impl<Fd>::call(std::forward<F>(f), std::forward<Args>(args)...));


template <typename AlwaysVoid, typename, typename...>
struct invoke_result_impl { };
template <typename F, typename...Args>
struct invoke_result_impl<decltype(void(INVOKE(std::declval<F>(), std::declval<Args>()...))), F, Args...>
{
    using type = decltype(INVOKE(std::declval<F>(), std::declval<Args>()...));
};

template <class F, class... ArgTypes>
struct invoke_result : invoke_result_impl<void, F, ArgTypes...> {};

template<class InputType, class BinaryFunction>
struct match_result_type
{
    using type = typename invoke_result<BinaryFunction, InputType, InputType>::type;
};

}
END_ROCPRIM_NAMESPACE

#endif
