

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_LOAD_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_LOAD_HPP_

#include "../config.hpp"
#include "../util_type.cuh"

BEGIN_HIPCUB_NAMESPACE

enum CacheLoadModifier : int32_t
{
    LOAD_DEFAULT,
    LOAD_CA,
    LOAD_CG,
    LOAD_CS,
    LOAD_CV,
    LOAD_LDG,
    LOAD_VOLATILE,
};

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T AsmThreadLoad(void * ptr)
{
    T retval = 0;
    __builtin_memcpy(&retval, ptr, sizeof(T));
    return retval;
}

#ifndef HIPCUB_THREAD_LOAD_USE_CACHE_MODIFIERS
#define HIPCUB_THREAD_LOAD_USE_CACHE_MODIFIERS 1
#endif

#if HIPCUB_THREAD_LOAD_USE_CACHE_MODIFIERS == 1



#define HIPCUB_ASM_THREAD_LOAD(cache_modifier,                                                                \
                               llvm_cache_modifier,                                                           \
                               type,                                                                          \
                               interim_type,                                                                  \
                               asm_operator,                                                                  \
                               output_modifier,                                                               \
                               wait_cmd)                                                                      \
    template<>                                                                                                \
    HIPCUB_DEVICE __forceinline__ type AsmThreadLoad<cache_modifier, type>(void * ptr)                        \
    {                                                                                                         \
        interim_type retval;                                                                                  \
        asm volatile(                                                                                         \
            #asm_operator " %0, %1 " llvm_cache_modifier "\n"                                                 \
            "\ts_waitcnt " wait_cmd "(0)" : "=" #output_modifier(retval) : "v"(ptr)                            \
        );                                                                                                    \
        return retval;                                                                                        \
    }


#define HIPCUB_ASM_THREAD_LOAD_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                  \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_load_sbyte, v, wait_cmd);      \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_load_sshort, v, wait_cmd);    \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_load_ubyte, v, wait_cmd);    \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_load_ushort, v, wait_cmd);  \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_load_dword, v, wait_cmd);   \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_load_dword, v, wait_cmd);      \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_load_dwordx2, v, wait_cmd); \
    HIPCUB_ASM_THREAD_LOAD(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_load_dwordx2, v, wait_cmd);

HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CA, "glc", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CG, "glc slc", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CV, "glc", "vmcnt");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_VOLATILE, "glc", "vmcnt");


HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_LDG, "", "");
HIPCUB_ASM_THREAD_LOAD_GROUP(LOAD_CS, "", "");

#endif

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename InputIteratorT>
HIPCUB_DEVICE __forceinline__
typename std::iterator_traits<InputIteratorT>::value_type ThreadLoad(InputIteratorT itr)
{
    using T  = typename std::iterator_traits<InputIteratorT>::value_type;
    T retval = ThreadLoad<MODIFIER>(&(*itr));
    return retval;
}

template<CacheLoadModifier MODIFIER = LOAD_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ T
ThreadLoad(T * ptr)
{
    return AsmThreadLoad<MODIFIER, T>(ptr);
}

END_HIPCUB_NAMESPACE
#endif
