

#ifndef ROCPRIM_THREAD_THREAD_LOAD_HPP_
#define ROCPRIM_THREAD_THREAD_LOAD_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

enum cache_load_modifier : int
{
    load_default,
    load_ca,
    load_cg,
    load_cs,
    load_cv,
    load_ldg,
    load_volatile,
};

namespace detail
{

template<cache_load_modifier MODIFIER = load_default, typename T>
ROCPRIM_DEVICE __forceinline__ T AsmThreadLoad(void * ptr)
{
    T retval = 0;
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

#if ROCPRIM_THREAD_LOAD_USE_CACHE_MODIFIERS == 1



#define ROCPRIM_ASM_THREAD_LOAD(cache_modifier,                                                                \
                               llvm_cache_modifier,                                                           \
                               type,                                                                          \
                               interim_type,                                                                  \
                               asm_operator,                                                                  \
                               output_modifier,                                                               \
                               wait_cmd)                                                                      \
    template<>                                                                                                \
    ROCPRIM_DEVICE __forceinline__ type AsmThreadLoad<cache_modifier, type>(void * ptr)                        \
    {                                                                                                         \
        interim_type retval;                                                                                  \
        asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier : "=" #output_modifier(retval) : "v"(ptr)); \
        asm volatile("s_waitcnt " wait_cmd "(%0)" : : "I"(0x00));                                             \
        return retval;                                                                                        \
    }


#define ROCPRIM_ASM_THREAD_LOAD_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                  \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_load_sbyte, v, wait_cmd);      \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_load_sshort, v, wait_cmd);    \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_load_ubyte, v, wait_cmd);    \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_load_ushort, v, wait_cmd);  \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_load_dword, v, wait_cmd);   \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_load_dword, v, wait_cmd);      \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_load_dwordx2, v, wait_cmd); \
    ROCPRIM_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_load_dwordx2, v, wait_cmd);


#ifndef __HIP_CPU_RT__
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ca, "glc", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cg, "glc slc", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cv, "glc", "vmcnt");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_volatile, "glc", "vmcnt");


ROCPRIM_ASM_THREAD_LOAD_GROUP(load_ldg, "", "");
ROCPRIM_ASM_THREAD_LOAD_GROUP(load_cs, "", "");
#endif

#endif

}






template <
    cache_load_modifier MODIFIER = load_default,
    typename InputIteratorT>
ROCPRIM_DEVICE ROCPRIM_INLINE
typename std::iterator_traits<InputIteratorT>::value_type
thread_load(InputIteratorT itr)
{
    using T = typename std::iterator_traits<InputIteratorT>::value_type;
    T retval = thread_load<MODIFIER>(&(*itr));
    return *itr;
}






template <
    cache_load_modifier MODIFIER = load_default,
    typename T>
ROCPRIM_DEVICE ROCPRIM_INLINE
T thread_load(T* ptr)
{
#ifndef __HIP_CPU_RT__
    return detail::AsmThreadLoad<MODIFIER, T>(ptr);
#else
    T retval;
    std::memcpy(&retval, ptr, sizeof(T));
    return retval;
#endif
}

END_ROCPRIM_NAMESPACE

#endif
