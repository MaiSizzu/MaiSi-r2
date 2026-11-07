



















#ifndef ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_
#define ROCPRIM_INTRINSICS_WARP_SHUFFLE_HPP_

#include <type_traits>

#include "../config.hpp"
#include "thread.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#ifdef __HIP_CPU_RT__


template <class To, class From>
typename std::enable_if_t<
    sizeof(To) == sizeof(From) &&
    std::is_trivially_copyable_v<From> &&
    std::is_trivially_copyable_v<To>,
    To>

bit_cast(const From& src) noexcept
{
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}
#endif

template<class T, class ShuffleOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    struct V { int words[words_no]; };
#ifdef __HIP_CPU_RT__
    V a = bit_cast<V>(input);
#else
    V a = __builtin_bit_cast(V, input);
#endif

    ROCPRIM_UNROLL
    for(int i = 0; i < words_no; i++)
    {
        a.words[i] = op(a.words[i]);
    }

#ifdef __HIP_CPU_RT__
    return bit_cast<T>(a);
#else
    return __builtin_bit_cast(T, a);
#endif
}

template<class T, class ShuffleOp>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::enable_if<!(std::is_trivially_copyable<T>::value && (sizeof(T) % sizeof(int) == 0)), T>::type
warp_shuffle_op(const T& input, ShuffleOp&& op)
{
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    T output;
    ROCPRIM_UNROLL
    for(int i = 0; i < words_no; i++)
    {
        const size_t s = (sizeof(int) < sizeof(T) - i * sizeof(int)) ? sizeof(int) : (sizeof(T) - i * sizeof(int));
        int word;
#ifdef __HIP_CPU_RT__
        std::memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), s);
#else
        __builtin_memcpy(&word, reinterpret_cast<const char*>(&input) + i * sizeof(int), s);
#endif
        word = op(word);
#ifdef __HIP_CPU_RT__
        std::memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, s);
#else
        __builtin_memcpy(reinterpret_cast<char*>(&output) + i * sizeof(int), &word, s);
#endif
    }

    return output;

}

template<class T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = false>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_move_dpp(const T& input)
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {






#if !defined(__HIP_CPU_RT__)
            return ::__builtin_amdgcn_mov_dpp(v, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
#else
            return v;
#endif
        }
    );
}







template<class T, int mask>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_swizzle(const T& input)
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return ::__builtin_amdgcn_ds_swizzle(v, mask);
        }
    );
}

}















template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle(const T& input, const int src_lane, const int width = device_warp_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl(v, src_lane, width);
        }
    );
}













template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_up(const T& input, const unsigned int delta, const int width = device_warp_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_up(v, delta, width);
        }
    );
}













template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_down(const T& input, const unsigned int delta, const int width = device_warp_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_down(v, delta, width);
        }
    );
}












template<class T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T warp_shuffle_xor(const T& input, const int lane_mask, const int width = device_warp_size())
{
    return detail::warp_shuffle_op(
        input,
        [=](int v) -> int
        {
            return __shfl_xor(v, lane_mask, width);
        }
    );
}

END_ROCPRIM_NAMESPACE

#endif



