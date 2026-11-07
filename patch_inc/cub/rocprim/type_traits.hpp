



















#ifndef ROCPRIM_TYPE_TRAITS_HPP_
#define ROCPRIM_TYPE_TRAITS_HPP_

#include <type_traits>


#include "config.hpp"
#include "types.hpp"




BEGIN_ROCPRIM_NAMESPACE



template<class T>
struct is_floating_point
    : std::integral_constant<
        bool,
        std::is_floating_point<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};


template<class T>
using is_integral = std::is_integral<T>;



template<class T>
struct is_arithmetic
    : std::integral_constant<
        bool,
        std::is_arithmetic<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};



template<class T>
struct is_fundamental
  : std::integral_constant<
        bool,
        std::is_fundamental<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
> {};


template<class T>
using is_unsigned = std::is_unsigned<T>;



template<class T>
struct is_signed
    : std::integral_constant<
        bool,
        std::is_signed<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};



template<class T>
struct is_scalar
    : std::integral_constant<
        bool,
        std::is_scalar<T>::value ||
        std::is_same<::rocprim::half, typename std::remove_cv<T>::type>::value ||
        std::is_same<::rocprim::bfloat16, typename std::remove_cv<T>::type>::value
    > {};



template<class T>
struct is_compound
    : std::integral_constant<
        bool,
        !is_fundamental<T>::value
    > {};

template<typename T, int size = 0>
struct get_unsigned_bits_type
{
  typedef typename get_unsigned_bits_type<T,sizeof(T)>::unsigned_type unsigned_type;
};

template<typename T>
struct get_unsigned_bits_type<T,1>
{
  typedef uint8_t unsigned_type;
};


template<typename T>
struct get_unsigned_bits_type<T,2>
{
  typedef uint16_t unsigned_type;
};


template<typename T>
struct get_unsigned_bits_type<T,4>
{
  typedef uint32_t unsigned_type;
};


template<typename T>
struct get_unsigned_bits_type<T,8>
{
  typedef uint64_t unsigned_type;
};

template<typename T, typename UnsignedBits>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleIn(UnsignedBits key)
    -> typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
  static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
  UnsignedBits mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
  return key ^ mask;
}

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleIn(UnsignedBits key)
    -> typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key ;
};

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleIn(UnsignedBits key)
    -> typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};

template<typename T, typename UnsignedBits>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleOut(UnsignedBits key)
    -> typename std::enable_if<is_floating_point<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    UnsignedBits mask = (key & HIGH_BIT) ? HIGH_BIT : UnsignedBits(-1);
    return key ^ mask;
}

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleOut(UnsignedBits key)
    -> typename std::enable_if<is_unsigned<T>::value, UnsignedBits>::type
{
    return key;
};

template<typename T, typename UnsignedBits>
static ROCPRIM_DEVICE ROCPRIM_INLINE
auto TwiddleOut(UnsignedBits key)
    -> typename std::enable_if<is_integral<T>::value && is_signed<T>::value, UnsignedBits>::type
{
    static const UnsignedBits   HIGH_BIT    = UnsignedBits(1) << ((sizeof(UnsignedBits) * 8) - 1);
    return key ^ HIGH_BIT;
};


END_ROCPRIM_NAMESPACE




#endif
