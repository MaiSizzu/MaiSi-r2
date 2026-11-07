



















#ifndef ROCPRIM_FUNCTIONAL_HPP_
#define ROCPRIM_FUNCTIONAL_HPP_

#include <functional>


#include "config.hpp"

BEGIN_ROCPRIM_NAMESPACE




#define ROCPRIM_PRINT_ERROR_ONCE(message) \
{                                          \
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x); \
    idx += threadIdx.y + (blockIdx.y * blockDim.y);             \
    idx += threadIdx.z + (blockIdx.z * blockDim.z);             \
    if (idx == 0)                                                        \
        printf("%s\n", #message);                                        \
}

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T max(const T& a, const T& b)
{
    return a < b ? b : a;
}

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T min(const T& a, const T& b)
{
    return a < b ? a : b;
}

template<class T>
ROCPRIM_HOST_DEVICE inline
void swap(T& a, T& b)
{
    T c = a;
    a = b;
    b = c;
}

template<class T = void>
struct less
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a < b;
    }
};

template<>
struct less<void>
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const U& b) const
    {
        return a < b;
    }
};

template<class T = void>
struct less_equal
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

template<>
struct less_equal<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

template<class T = void>
struct greater
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

template<>
struct greater<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

template<class T = void>
struct greater_equal
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

template<>
struct greater_equal<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

template<class T = void>
struct equal_to
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

template<>
struct equal_to<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

template<class T = void>
struct not_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template<>
struct not_equal_to<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

static __forceinline__ __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
    return make_uint2(a.x + b.x, a.y + b.y);
}

template<class T = void>
struct plus
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<>
struct plus<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<class T = void>
struct minus
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

template<>
struct minus<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

template<class T = void>
struct multiplies
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

template<>
struct multiplies<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

template<class T = void>
struct maximum
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

template<>
struct maximum<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

template<class T = void>
struct minimum
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

template<>
struct minimum<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

template<class T = void>
struct identity
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a) const
    {
        return a;
    }
};

template<>
struct identity<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a) const
    {
        return a;
    }
};


template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{

    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
    enum {VALUE = (1 << (COUNT - 1) < N) ?
        COUNT :
        COUNT - 1 };
};

#endif




template <typename A, typename B>
struct Equals
{
    enum {
        VALUE = 0,
        NEGATE = 1
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename A>
struct Equals <A, A>
{
    enum {
        VALUE = 1,
        NEGATE = 0
    };
};

#endif

template <int A>
struct Int2Type
{
   enum {VALUE = A};
};




END_ROCPRIM_NAMESPACE

#endif
