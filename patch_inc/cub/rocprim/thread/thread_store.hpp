

#ifndef ROCPRIM_THREAD_THREAD_STORE_HPP_
#define ROCPRIM_THREAD_THREAD_STORE_HPP_


#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

enum cache_store_modifier
{
    store_default,
    store_wb,
    store_cg,
    store_cs,
    store_wt,
    store_volatile,
};

namespace detail
{

template<cache_store_modifier MODIFIER = store_default, typename T>
ROCPRIM_DEVICE __forceinline__ void AsmThreadStore(void * ptr, T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

#if ROCPRIM_THREAD_STORE_USE_CACHE_MODIFIERS == 1






#define ROCPRIM_ASM_THREAD_STORE(cache_modifier,                                                              \
                                llvm_cache_modifier,                                                         \
                                type,                                                                        \
                                interim_type,                                                                \
                                asm_operator,                                                                \
                                output_modifier,                                                             \
                                wait_cmd)                                                                    \
    template<>                                                                                               \
    ROCPRIM_DEVICE __forceinline__ void AsmThreadStore<cache_modifier, type>(void * ptr, type val)            \
    {                                                                                                        \
        interim_type temp_val = val;                                                          \
        asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier : : "v"(ptr), #output_modifier(temp_val)); \
        asm volatile("s_waitcnt " wait_cmd "(%0)" : : "I"(0x00));                                            \
    }



#define ROCPRIM_ASM_THREAD_STORE_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_store_byte, v, wait_cmd);       \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_store_short, v, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_store_byte, v, wait_cmd);     \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_store_short, v, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_store_dword, v, wait_cmd);   \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_store_dword, v, wait_cmd);      \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_store_dwordx2, v, wait_cmd); \
    ROCPRIM_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_store_dwordx2, v, wait_cmd);


#ifndef __HIP_CPU_RT__
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wb, "glc", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_cg, "glc slc", "");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_wt, "glc", "vmcnt");
ROCPRIM_ASM_THREAD_STORE_GROUP(store_volatile, "glc", "vmcnt");


ROCPRIM_ASM_THREAD_STORE_GROUP(store_cs, "", "");
#endif

#endif

}







template <
    cache_store_modifier MODIFIER = store_default,
    typename OutputIteratorT,
    typename T
>
ROCPRIM_DEVICE ROCPRIM_INLINE void thread_store(
    OutputIteratorT itr,
    T               val)
{
    thread_store<MODIFIER>(&(*itr), val);
}






template <
    cache_store_modifier MODIFIER = store_default,
    typename T
>
ROCPRIM_DEVICE ROCPRIM_INLINE void thread_store(
    T *ptr,
    T val)
{
#ifndef __HIP_CPU_RT__
    detail::AsmThreadStore<MODIFIER, T>(ptr, val);
#else
    std::memcpy(ptr, &val, sizeof(T));
#endif
}

END_ROCPRIM_NAMESPACE

#endif
