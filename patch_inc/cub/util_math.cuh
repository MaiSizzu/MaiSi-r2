

#ifndef HIPCUB_ROCPRIM_UTIL_MATH_HPP_
#define HIPCUB_ROCPRIM_UTIL_MATH_HPP_



BEGIN_HIPCUB_NAMESPACE


template <typename T>
constexpr __device__ __host__ T MidPoint(T begin, T end)
{
    return begin + (end - begin) / 2;
}

END_HIPCUB_NAMESPACE

#endif
