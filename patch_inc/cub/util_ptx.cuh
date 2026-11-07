

#ifndef HIPCUB_ROCPRIM_UTIL_PTX_HPP_
#define HIPCUB_ROCPRIM_UTIL_PTX_HPP_

#include <cstdint>
#include <type_traits>

#include "config.hpp"

#include <cub/rocprim/intrinsics/warp_shuffle.hpp>

#define HIPCUB_WARP_THREADS ::rocprim::warp_size()
#define HIPCUB_DEVICE_WARP_THREADS ::rocprim::device_warp_size()
#define HIPCUB_HOST_WARP_THREADS ::rocprim::host_warp_size()
#define HIPCUB_ARCH 1


BEGIN_HIPCUB_NAMESPACE


















HIPCUB_DEVICE inline
int RowMajorTid(int block_dim_x, int block_dim_y, int block_dim_z)
{
    return ((block_dim_z == 1) ? 0 : (threadIdx.z * block_dim_x * block_dim_y))
        + ((block_dim_y == 1) ? 0 : (threadIdx.y * block_dim_x))
        + threadIdx.x;
}

HIPCUB_DEVICE inline
unsigned int LaneId()
{
    return ::rocprim::lane_id();
}

HIPCUB_DEVICE inline
unsigned int WarpId()
{
    return ::rocprim::warp_id();
}

template <int LOGICAL_WARP_THREADS, int  = 0>
HIPCUB_DEVICE inline 
uint64_t WarpMask(unsigned int warp_id) {
    constexpr bool is_pow_of_two = ::rocprim::detail::is_power_of_two(LOGICAL_WARP_THREADS);
    constexpr bool is_arch_warp =
        LOGICAL_WARP_THREADS == ::rocprim::device_warp_size();

    uint64_t member_mask = uint64_t(-1) >> (64 - LOGICAL_WARP_THREADS);

    if (is_pow_of_two && !is_arch_warp) {
        member_mask <<= warp_id * LOGICAL_WARP_THREADS;
    }

    return member_mask;
}


HIPCUB_DEVICE inline
uint64_t LaneMaskLt()
{
    return (uint64_t(1) << LaneId()) - 1;
}


HIPCUB_DEVICE inline
uint64_t LaneMaskLe()
{
    return ((uint64_t(1) << LaneId()) << 1) - 1;
}


HIPCUB_DEVICE inline
uint64_t LaneMaskGt()
{
    return uint64_t(-1)^LaneMaskLe();
}


HIPCUB_DEVICE inline
uint64_t LaneMaskGe()
{
    return uint64_t(-1)^LaneMaskLt();
}



template <
    int LOGICAL_WARP_THREADS,
    typename T
>
HIPCUB_DEVICE inline
T ShuffleUp(T input,
            int src_offset,
            int first_thread,
            unsigned int member_mask)
{

    (void) first_thread;


    (void) member_mask;
    return ::rocprim::warp_shuffle_up(
        input, src_offset, LOGICAL_WARP_THREADS
    );
}

template <
    int LOGICAL_WARP_THREADS,
    typename T
>
HIPCUB_DEVICE inline
T ShuffleDown(T input,
              int src_offset,
              int last_thread,
              unsigned int member_mask)
{

    (void) last_thread;


    (void) member_mask;
    return ::rocprim::warp_shuffle_down(
        input, src_offset, LOGICAL_WARP_THREADS
    );
}

template <
    int LOGICAL_WARP_THREADS,
    typename T
>
HIPCUB_DEVICE inline
T ShuffleIndex(T input,
               int src_lane,
               unsigned int member_mask)
{


    (void) member_mask;
    return ::rocprim::warp_shuffle(
        input, src_lane, LOGICAL_WARP_THREADS
    );
}



HIPCUB_DEVICE inline
unsigned int SHR_ADD(unsigned int x,
                     unsigned int shift,
                     unsigned int addend)
{
    return (x >> shift) + addend;
}

HIPCUB_DEVICE inline
unsigned int SHL_ADD(unsigned int x,
                     unsigned int shift,
                     unsigned int addend)
{
    return (x << shift) + addend;
}

namespace detail {

template <typename UnsignedBits>
HIPCUB_DEVICE inline
auto unsigned_bit_extract(UnsignedBits source,
                          unsigned int bit_start,
                          unsigned int num_bits)
    -> typename std::enable_if<sizeof(UnsignedBits) == 8, unsigned int>::type
{
    #ifdef __CUDACC__
        return __bitextract_u64(source, bit_start, num_bits);
    #else
        return (source << (64 - bit_start - num_bits)) >> (64 - num_bits);
    #endif
}

template <typename UnsignedBits>
HIPCUB_DEVICE inline
auto unsigned_bit_extract(UnsignedBits source,
                          unsigned int bit_start,
                          unsigned int num_bits)
    -> typename std::enable_if<sizeof(UnsignedBits) < 8, unsigned int>::type
{
    #ifdef __CUDACC__
        return __bitextract_u32(source, bit_start, num_bits);
    #else
        return (static_cast<unsigned int>(source) << (32 - bit_start - num_bits)) >> (32 - num_bits);
    #endif
}

}




template <typename UnsignedBits>
HIPCUB_DEVICE inline
unsigned int BFE(UnsignedBits source,
                 unsigned int bit_start,
                 unsigned int num_bits)
{
    static_assert(std::is_unsigned<UnsignedBits>::value, "UnsignedBits must be unsigned");
    return detail::unsigned_bit_extract(source, bit_start, num_bits);
}



HIPCUB_DEVICE inline
void BFI(unsigned int &ret,
         unsigned int x,
         unsigned int y,
         unsigned int bit_start,
         unsigned int num_bits)
{
    #ifdef __CUDACC__
        ret = __bitinsert_u32(x, y, bit_start, num_bits);
    #else
        x <<= bit_start;
        unsigned int MASK_X = ((1 << num_bits) - 1) << bit_start;
        unsigned int MASK_Y = ~MASK_X;
        ret = (y & MASK_Y) | (x & MASK_X);
    #endif
}

HIPCUB_DEVICE inline
unsigned int IADD3(unsigned int x, unsigned int y, unsigned int z)
{
    return x + y + z;
}

HIPCUB_DEVICE inline
int PRMT(unsigned int a, unsigned int b, unsigned int index)
{
    return ::__byte_perm(a, b, index);
}

HIPCUB_DEVICE inline
void BAR(int count)
{
    (void) count;
    __syncthreads();
}

HIPCUB_DEVICE inline
void CTA_SYNC()
{
    __syncthreads();
}

HIPCUB_DEVICE inline
void WARP_SYNC(unsigned int member_mask)
{
    (void) member_mask;
    ::rocprim::wave_barrier();
}

HIPCUB_DEVICE inline
int WARP_ANY(int predicate, uint64_t member_mask)
{
    (void) member_mask;
    return ::__any(predicate);
}

HIPCUB_DEVICE inline
int WARP_ALL(int predicate, uint64_t member_mask)
{
    (void) member_mask;
    return ::__all(predicate);
}

HIPCUB_DEVICE inline
int64_t WARP_BALLOT(int predicate, uint64_t member_mask)
{
    (void) member_mask;
    return __ballot(predicate);
}

END_HIPCUB_NAMESPACE

#endif
