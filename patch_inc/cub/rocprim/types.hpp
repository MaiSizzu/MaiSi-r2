



















#ifndef ROCPRIM_TYPES_HPP_
#define ROCPRIM_TYPES_HPP_

#include <type_traits>


#include "config.hpp"

#include "types/future_value.hpp"
#include "types/double_buffer.hpp"
#include "types/integer_sequence.hpp"
#include "types/key_value_pair.hpp"
#include "types/tuple.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{



#ifndef _MSC_VER
#define DEFINE_VECTOR_TYPE(name, base) \
\
struct alignas(sizeof(base) * 2) name##2 \
{ \
    typedef base vector_value_type __attribute__((ext_vector_type(2))); \
    union { \
        vector_value_type data; \
        struct { base x, y; }; \
    }; \
}; \
\
struct alignas(sizeof(base) * 4) name##4 \
{ \
    typedef base vector_value_type __attribute__((ext_vector_type(4))); \
    union { \
        vector_value_type data; \
        struct { base x, y, w, z; }; \
    }; \
};
#else
#define DEFINE_VECTOR_TYPE(name, base) \
\
struct alignas(sizeof(base) * 2) name##2 \
{ \
    typedef base vector_value_type; \
    union { \
        vector_value_type data; \
        struct { base x, y; }; \
    }; \
}; \
\
struct alignas(sizeof(base) * 4) name##4 \
{ \
    typedef base vector_value_type; \
    union { \
        vector_value_type data; \
        struct { base x, y, w, z; }; \
    }; \
};
#endif

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4201 )
#endif
DEFINE_VECTOR_TYPE(char, char);
DEFINE_VECTOR_TYPE(short, short);
DEFINE_VECTOR_TYPE(int, int);
DEFINE_VECTOR_TYPE(longlong, long long);
#ifdef _MSC_VER
#pragma warning( pop )
#endif

template <class T, unsigned int NumElements>
struct make_vector_type
{
    using type = void;
};

#define DEFINE_MAKE_VECTOR_N_TYPE(name, base, suffix) \
template<> \
struct make_vector_type<base, suffix> \
{ \
    using type = name##suffix; \
};

#define DEFINE_MAKE_VECTOR_TYPE(name, base) \
\
template <> \
struct make_vector_type<base, 1> \
{ \
    using type = base; \
}; \
DEFINE_MAKE_VECTOR_N_TYPE(name, base, 2) \
DEFINE_MAKE_VECTOR_N_TYPE(name, base, 4)

DEFINE_MAKE_VECTOR_TYPE(char, char);
DEFINE_MAKE_VECTOR_TYPE(short, short);
DEFINE_MAKE_VECTOR_TYPE(int, int);
DEFINE_MAKE_VECTOR_TYPE(longlong, long long);

#undef DEFINE_VECTOR_TYPE
#undef DEFINE_MAKE_VECTOR_TYPE
#undef DEFINE_MAKE_VECTOR_N_TYPE

}



struct empty_type {};



struct empty_binary_op
{
    constexpr empty_type operator()(const empty_type&, const empty_type&) const { return empty_type{}; }
};


using half = ::__half;

using bfloat16 = ::__nv_bfloat16;


#ifndef __AMDGCN_WAVEFRONT_SIZE


#define __AMDGCN_WAVEFRONT_SIZE 64
#endif
#if __AMDGCN_WAVEFRONT_SIZE == 32
using lane_mask_type = unsigned int;
#elif __AMDGCN_WAVEFRONT_SIZE == 64
using lane_mask_type = unsigned long long int;
#endif

#ifdef __HIP_CPU_RT__
using native_half = half;
#else
using native_half = _Float16;
#endif

#ifdef __HIP_CPU_RT__

using native_bfloat16 = bfloat16;
#else
using native_bfloat16 = bfloat16;
#endif

END_ROCPRIM_NAMESPACE




#endif
