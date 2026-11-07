



 #ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_
 #define HIPCUB_ROCPRIM_BLOCK_BLOCK_RADIX_RANK_HPP_

#include <stdint.h>

#include "../config.hpp"
#include "../util_type.cuh"
#include "../util_ptx.cuh"

#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../block/block_scan.cuh"
#include "../block/radix_rank_sort_operations.cuh"

BEGIN_HIPCUB_NAMESPACE




template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    bool                    MEMOIZE_OUTER_SCAN   = false,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM = BLOCK_SCAN_WARP_SCANS,
    cudaSharedMemConfig      SMEM_CONFIG          = cudaSharedMemBankSizeFourByte,
    int                     BLOCK_DIM_Y          = 1,
    int                     BLOCK_DIM_Z          = 1,
    int                     ARCH                 = HIPCUB_ARCH >
class BlockRadixRank
{
private:

    


    typedef unsigned short DigitCounter;


    typedef typename std::conditional<(SMEM_CONFIG == cudaSharedMemBankSizeEightByte),
        unsigned long long,
        unsigned int>::type PackedCounter;

    enum
    {

        BLOCK_THREADS               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        RADIX_DIGITS                = 1 << RADIX_BITS,

        LOG_WARP_THREADS            = Log2<ARCH>::VALUE,
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                       = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        BYTES_PER_COUNTER           = sizeof(DigitCounter),
        LOG_BYTES_PER_COUNTER       = Log2<BYTES_PER_COUNTER>::VALUE,

        PACKING_RATIO               = sizeof(PackedCounter) / sizeof(DigitCounter),
        LOG_PACKING_RATIO           = Log2<PACKING_RATIO>::VALUE,

        LOG_COUNTER_LANES           = rocprim::maximum<int>()((int(RADIX_BITS) - int(LOG_PACKING_RATIO)), 0),
        COUNTER_LANES               = 1 << LOG_COUNTER_LANES,


        PADDED_COUNTER_LANES        = COUNTER_LANES + 1,
        RAKING_SEGMENT              = PADDED_COUNTER_LANES,
    };

public:

    enum
    {

        BINS_TRACKED_PER_THREAD = rocprim::maximum<int>()(1, (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS),
    };

private:



    typedef BlockScan<
            PackedCounter,
            BLOCK_DIM_X,
            INNER_SCAN_ALGORITHM,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z,
            ARCH>
        BlockScan;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


    struct __align__(16) _TempStorage
    {
        union Aliasable
        {
            DigitCounter            digit_counters[PADDED_COUNTER_LANES * BLOCK_THREADS * PACKING_RATIO];
            PackedCounter           raking_grid[BLOCK_THREADS * RAKING_SEGMENT];

        } aliasable;


        typename BlockScan::TempStorage block_scan;
    };

#endif

    


    _TempStorage &temp_storage;


    unsigned int linear_tid;


    PackedCounter cached_segment[RAKING_SEGMENT];


    

    
    HIPCUB_DEVICE inline _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    
    HIPCUB_DEVICE inline PackedCounter Upsweep()
    {
        PackedCounter *smem_raking_ptr = &temp_storage.aliasable.raking_grid[linear_tid * RAKING_SEGMENT];
        PackedCounter *raking_ptr;

        if (MEMOIZE_OUTER_SCAN)
        {

            #pragma unroll
            for (int i = 0; i < RAKING_SEGMENT; i++)
            {
                cached_segment[i] = smem_raking_ptr[i];
            }
            raking_ptr = cached_segment;
        }
        else
        {
            raking_ptr = smem_raking_ptr;
        }

        return internal::ThreadReduce<RAKING_SEGMENT>(raking_ptr, Sum());
    }



    HIPCUB_DEVICE inline void ExclusiveDownsweep(
        PackedCounter raking_partial)
    {
        PackedCounter *smem_raking_ptr = &temp_storage.aliasable.raking_grid[linear_tid * RAKING_SEGMENT];

        PackedCounter *raking_ptr = (MEMOIZE_OUTER_SCAN) ?
            cached_segment :
            smem_raking_ptr;


        internal::ThreadScanExclusive<RAKING_SEGMENT>(raking_ptr, raking_ptr, Sum(), raking_partial);

        if (MEMOIZE_OUTER_SCAN)
        {

            #pragma unroll
            for (int i = 0; i < RAKING_SEGMENT; i++)
            {
                smem_raking_ptr[i] = cached_segment[i];
            }
        }
    }


    
    HIPCUB_DEVICE inline void ResetCounters()
    {

        #pragma unroll
        for (int LANE = 0; LANE < PADDED_COUNTER_LANES; LANE++)
        {
            #pragma unroll
            for (int SUB_COUNTER = 0; SUB_COUNTER < PACKING_RATIO; SUB_COUNTER++)
            {
                temp_storage.aliasable.digit_counters[(LANE * BLOCK_THREADS + linear_tid) * PACKING_RATIO + SUB_COUNTER] = 0;
            }
        }
    }


    
    struct PrefixCallBack
    {
        HIPCUB_DEVICE inline PackedCounter operator()(PackedCounter block_aggregate)
        {
            PackedCounter block_prefix = 0;


            #pragma unroll
            for (int PACKED = 1; PACKED < PACKING_RATIO; PACKED++)
            {
                block_prefix += block_aggregate << (sizeof(DigitCounter) * 8 * PACKED);
            }

            return block_prefix;
        }
    };


    
    HIPCUB_DEVICE inline void ScanCounters()
    {

        PackedCounter raking_partial = Upsweep();


        PackedCounter exclusive_partial;
        PrefixCallBack prefix_call_back;
        BlockScan(temp_storage.block_scan).ExclusiveSum(raking_partial, exclusive_partial, prefix_call_back);


        ExclusiveDownsweep(exclusive_partial);
    }

public:


    struct TempStorage : Uninitialized<_TempStorage> {};


    


    
    HIPCUB_DEVICE inline BlockRadixRank()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}


    
    HIPCUB_DEVICE inline BlockRadixRank(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}



    


    
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    HIPCUB_DEVICE inline void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor)
    {
        DigitCounter    thread_prefixes[KEYS_PER_THREAD];
        DigitCounter*   digit_counters[KEYS_PER_THREAD];


        ResetCounters();

        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        {

            unsigned int digit = digit_extractor.Digit(keys[ITEM]);


            unsigned int sub_counter = digit >> LOG_COUNTER_LANES;


            unsigned int counter_lane = digit & (COUNTER_LANES - 1);

            if (IS_DESCENDING)
            {
                sub_counter = PACKING_RATIO - 1 - sub_counter;
                counter_lane = COUNTER_LANES - 1 - counter_lane;
            }


            digit_counters[ITEM] = &temp_storage.aliasable.digit_counters[counter_lane * BLOCK_THREADS * PACKING_RATIO + linear_tid * PACKING_RATIO + sub_counter];


            thread_prefixes[ITEM] = *digit_counters[ITEM];


            *digit_counters[ITEM] = thread_prefixes[ITEM] + 1;
        }

        ::rocprim::syncthreads();


        ScanCounters();

        ::rocprim::syncthreads();


        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        {

            ranks[ITEM] = thread_prefixes[ITEM] + *digit_counters[ITEM];
        }
    }


    
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    HIPCUB_DEVICE inline void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])
    {

        RankKeys(keys, ranks, digit_extractor);


        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                    bin_idx = RADIX_DIGITS - bin_idx - 1;



                unsigned int counter_lane   = (bin_idx & (COUNTER_LANES - 1));
                unsigned int sub_counter    = bin_idx >> (LOG_COUNTER_LANES);

                exclusive_digit_prefix[track] = temp_storage.aliasable.digit_counter[counter_lane * BLOCK_THREADS * PACKING_RATIO + sub_counter];
            }
        }
    }
};






template <
    int                     BLOCK_DIM_X,
    int                     RADIX_BITS,
    bool                    IS_DESCENDING,
    BlockScanAlgorithm      INNER_SCAN_ALGORITHM    = BLOCK_SCAN_WARP_SCANS,
    int                     BLOCK_DIM_Y             = 1,
    int                     BLOCK_DIM_Z             = 1,
    int                     ARCH                = HIPCUB_ARCH>
class BlockRadixRankMatch
{
private:

    

    typedef int32_t    RankT;
    typedef int32_t    DigitCounterT;

    enum
    {

        BLOCK_THREADS               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        RADIX_DIGITS                = 1 << RADIX_BITS,

        LOG_WARP_THREADS            = Log2<ARCH>::VALUE,
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                       = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        PADDED_WARPS            = ((WARPS & 0x1) == 0) ?
                                    WARPS + 1 :
                                    WARPS,

        COUNTERS                = PADDED_WARPS * RADIX_DIGITS,
        RAKING_SEGMENT          = (COUNTERS + BLOCK_THREADS - 1) / BLOCK_THREADS,
        PADDED_RAKING_SEGMENT   = ((RAKING_SEGMENT & 0x1) == 0) ?
                                    RAKING_SEGMENT + 1 :
                                    RAKING_SEGMENT,
    };

public:

    enum
    {

        BINS_TRACKED_PER_THREAD = rocprim::maximum<int>()(1, (RADIX_DIGITS + BLOCK_THREADS - 1) / BLOCK_THREADS),
    };

private:


    typedef BlockScan<
            DigitCounterT,
            BLOCK_THREADS,
            INNER_SCAN_ALGORITHM,
            BLOCK_DIM_Y,
            BLOCK_DIM_Z,
            ARCH>
        BlockScanT;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    struct __align__(16) _TempStorage
    {
        typename BlockScanT::TempStorage            block_scan;

        union __align__(16) Aliasable
        {
            volatile DigitCounterT                  warp_digit_counters[RADIX_DIGITS * PADDED_WARPS];
            DigitCounterT                           raking_grid[BLOCK_THREADS * PADDED_RAKING_SEGMENT];

        } aliasable;
    };
#endif

    


    _TempStorage &temp_storage;


    unsigned int linear_tid;



public:


    struct TempStorage : Uninitialized<_TempStorage> {};


    



    
    HIPCUB_DEVICE inline BlockRadixRankMatch(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {}



    


    
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor)
    {


        #pragma unroll
        for (int ITEM = 0; ITEM < PADDED_RAKING_SEGMENT; ++ITEM)
            temp_storage.aliasable.raking_grid[linear_tid * PADDED_RAKING_SEGMENT + ITEM] = 0;

        ::rocprim::syncthreads();



        volatile DigitCounterT  *digit_counters[KEYS_PER_THREAD];
        uint32_t                warp_id         = linear_tid >> LOG_WARP_THREADS;
        uint32_t                lane_mask_lt    = LaneMaskLt();

        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
        {

            uint32_t digit = digit_extractor.Digit(keys[ITEM]);

            if (IS_DESCENDING)
                digit = RADIX_DIGITS - digit - 1;


            uint32_t peer_mask = rocprim::MatchAny<RADIX_BITS>(digit);


            digit_counters[ITEM] = &temp_storage.aliasable.warp_digit_counters[digit * PADDED_WARPS + warp_id];


            DigitCounterT warp_digit_prefix = *digit_counters[ITEM];


            WARP_SYNC(0xFFFFFFFF);


            int32_t digit_count = __popc(peer_mask);


            int32_t peer_digit_prefix = __popc(peer_mask & lane_mask_lt);

            if (peer_digit_prefix == 0)
            {

                *digit_counters[ITEM] = DigitCounterT(warp_digit_prefix + digit_count);
            }


            WARP_SYNC(0xFFFFFFFF);


            ranks[ITEM] = warp_digit_prefix + DigitCounterT(peer_digit_prefix);
        }

        ::rocprim::syncthreads();



        DigitCounterT scan_counters[PADDED_RAKING_SEGMENT];

        #pragma unroll
        for (int ITEM = 0; ITEM < PADDED_RAKING_SEGMENT; ++ITEM)
            scan_counters[ITEM] = temp_storage.aliasable.raking_grid[linear_tid * PADDED_RAKING_SEGMENT + ITEM];

        BlockScanT(temp_storage.block_scan).ExclusiveSum(scan_counters, scan_counters);

        #pragma unroll
        for (int ITEM = 0; ITEM < PADDED_RAKING_SEGMENT; ++ITEM)
            temp_storage.aliasable.raking_grid[linear_tid * PADDED_RAKING_SEGMENT + ITEM] = scan_counters[ITEM];

        ::rocprim::syncthreads();


        #pragma unroll
        for (int ITEM = 0; ITEM < KEYS_PER_THREAD; ++ITEM)
            ranks[ITEM] += *digit_counters[ITEM];
    }


    
    template <
        typename        UnsignedBits,
        int             KEYS_PER_THREAD,
        typename        DigitExtractorT>
    __device__ __forceinline__ void RankKeys(
        UnsignedBits    (&keys)[KEYS_PER_THREAD],
        int             (&ranks)[KEYS_PER_THREAD],
        DigitExtractorT digit_extractor,
        int             (&exclusive_digit_prefix)[BINS_TRACKED_PER_THREAD])
    {
        RankKeys(keys, ranks, digit_extractor);


        #pragma unroll
        for (int track = 0; track < BINS_TRACKED_PER_THREAD; ++track)
        {
            int bin_idx = (linear_tid * BINS_TRACKED_PER_THREAD) + track;

            if ((BLOCK_THREADS == RADIX_DIGITS) || (bin_idx < RADIX_DIGITS))
            {
                if (IS_DESCENDING)
                    bin_idx = RADIX_DIGITS - bin_idx - 1;

                exclusive_digit_prefix[track] = temp_storage.aliasable.warp_digit_counters[bin_idx * PADDED_WARPS];
            }
        }
    }
};



END_HIPCUB_NAMESPACE

#endif
