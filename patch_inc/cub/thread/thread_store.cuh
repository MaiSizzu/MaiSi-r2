

#ifndef HIPCUB_ROCPRIM_THREAD_THREAD_STORE_HPP_
#define HIPCUB_ROCPRIM_THREAD_THREAD_STORE_HPP_
BEGIN_HIPCUB_NAMESPACE

enum CacheStoreModifier
{
    STORE_DEFAULT,
    STORE_WB,
    STORE_CG,
    STORE_CS,
    STORE_WT,
    STORE_VOLATILE,
};



template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename T>
HIPCUB_DEVICE __forceinline__ void AsmThreadStore(void * ptr, T val)
{
    __builtin_memcpy(ptr, &val, sizeof(T));
}

#if HIPCUB_THREAD_STORE_USE_CACHE_MODIFIERS == 1






#define HIPCUB_ASM_THREAD_STORE(cache_modifier,                                                              \
                                llvm_cache_modifier,                                                         \
                                type,                                                                        \
                                interim_type,                                                                \
                                asm_operator,                                                                \
                                output_modifier,                                                             \
                                wait_cmd)                                                                    \
    template<>                                                                                               \
    HIPCUB_DEVICE __forceinline__ void AsmThreadStore<cache_modifier, type>(void * ptr, type val)            \
    {                                                                                                        \
        interim_type temp_val = val;                                                          \
        asm volatile(#asm_operator " %0, %1 " llvm_cache_modifier : : "v"(ptr), #output_modifier(temp_val)); \
        asm volatile("s_waitcnt " wait_cmd "(%0)" : : "I"(0x00));                                            \
    }



#define HIPCUB_ASM_THREAD_STORE_GROUP(cache_modifier, llvm_cache_modifier, wait_cmd)                                   \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int8_t, int16_t, flat_store_byte, v, wait_cmd);       \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, int16_t, int16_t, flat_store_short, v, wait_cmd);     \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint8_t, uint16_t, flat_store_byte, v, wait_cmd);     \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint16_t, uint16_t, flat_store_short, v, wait_cmd);   \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint32_t, uint32_t, flat_store_dword, v, wait_cmd);   \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, float, uint32_t, flat_store_dword, v, wait_cmd);      \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, uint64_t, uint64_t, flat_store_dwordx2, v, wait_cmd); \
    HIPCUB_ASM_THREAD_STORE(cache_modifier, llvm_cache_modifier, double, uint64_t, flat_store_dwordx2, v, wait_cmd);

HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WB, "glc", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_CG, "glc slc", "");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_WT, "glc", "vmcnt");
HIPCUB_ASM_THREAD_STORE_GROUP(STORE_VOLATILE, "glc", "vmcnt");


HIPCUB_ASM_THREAD_STORE_GROUP(STORE_CS, "", "");

#endif

template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename OutputIteratorT, typename T>
__device__ __forceinline__ void ThreadStore(OutputIteratorT itr, T val)
{
    ThreadStore<MODIFIER>(&(*itr), val);
}

template<CacheStoreModifier MODIFIER = STORE_DEFAULT, typename T>
__device__ __forceinline__ void ThreadStore(T * ptr, T val)
{
    AsmThreadStore<MODIFIER, T>(ptr, val);
}

END_HIPCUB_NAMESPACE
#endif
