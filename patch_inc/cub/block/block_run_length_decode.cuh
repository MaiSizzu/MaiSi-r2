

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_RUN_LENGTH_DECODE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_RUN_LENGTH_DECODE_HPP_

#include "../config.hpp"
#include "../thread/thread_search.cuh"
#include "../util_math.cuh"
#include "../util_ptx.cuh"
#include "../util_type.cuh"
#include "block_scan.cuh"
#include <limits>
#include <type_traits>

BEGIN_HIPCUB_NAMESPACE


template <typename ItemT,
          int BLOCK_DIM_X,
          int RUNS_PER_THREAD,
          int DECODED_ITEMS_PER_THREAD,
          typename DecodedOffsetT = uint32_t,
          int BLOCK_DIM_Y = 1,
          int BLOCK_DIM_Z = 1>
class BlockRunLengthDecode
{



private:

    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;


    static constexpr int BLOCK_RUNS = BLOCK_THREADS * RUNS_PER_THREAD;


    using RunOffsetScanT = BlockScan<DecodedOffsetT, BLOCK_DIM_X, BLOCK_SCAN_WARP_SCANS, BLOCK_DIM_Y, BLOCK_DIM_Z>;


    using RunOffsetT = uint32_t;


    union _TempStorage
    {
        typename RunOffsetScanT::TempStorage offset_scan;
        struct
        {
            ItemT run_values[BLOCK_RUNS];
            DecodedOffsetT run_offsets[BLOCK_RUNS];
        } runs;
    };


    HIPCUB_DEVICE __forceinline__ _TempStorage &PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }


    _TempStorage &temp_storage;


    uint32_t linear_tid;

public:
    struct TempStorage : Uninitialized<_TempStorage>
    {
    };





    
    template <typename RunLengthT, typename TotalDecodedSizeT>
    HIPCUB_DEVICE __forceinline__ BlockRunLengthDecode(TempStorage &temp_storage,
                                                       ItemT (&run_values)[RUNS_PER_THREAD],
                                                       RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                                       TotalDecodedSizeT &total_decoded_size)
        : temp_storage(temp_storage.Alias()), linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {
        InitWithRunLengths(run_values, run_lengths, total_decoded_size);
    }

    
    template <typename UserRunOffsetT>
    HIPCUB_DEVICE __forceinline__ BlockRunLengthDecode(TempStorage &temp_storage,
                                                       ItemT (&run_values)[RUNS_PER_THREAD],
                                                       UserRunOffsetT (&run_offsets)[RUNS_PER_THREAD])
        : temp_storage(temp_storage.Alias()), linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {
        InitWithRunOffsets(run_values, run_offsets);
    }

    
    template <typename RunLengthT, typename TotalDecodedSizeT>
    HIPCUB_DEVICE __forceinline__ BlockRunLengthDecode(ItemT (&run_values)[RUNS_PER_THREAD],
                                                       RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                                       TotalDecodedSizeT &total_decoded_size)
        : temp_storage(PrivateStorage()), linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {
        InitWithRunLengths(run_values, run_lengths, total_decoded_size);
    }

    
    template <typename UserRunOffsetT>
    HIPCUB_DEVICE __forceinline__ BlockRunLengthDecode(ItemT (&run_values)[RUNS_PER_THREAD],
                                                    UserRunOffsetT (&run_offsets)[RUNS_PER_THREAD])
        : temp_storage(PrivateStorage()), linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
    {
        InitWithRunOffsets(run_values, run_offsets);
    }

private:
    
    template <int MAX_NUM_ITEMS,
              typename InputIteratorT,
              typename OffsetT,
              typename T>
    HIPCUB_DEVICE __forceinline__ OffsetT StaticUpperBound(InputIteratorT input,
                                                           OffsetT num_items,
                                                           T val)
    {
        OffsetT lower_bound = 0;
        OffsetT upper_bound = num_items;
        #pragma unroll
        for (int i = 0; i <= Log2<MAX_NUM_ITEMS>::VALUE; i++)
        {
            OffsetT mid = cub::MidPoint<OffsetT>(lower_bound, upper_bound);
            mid = (rocprim::min)(mid, num_items - 1);

            if (val < input[mid])
            {
                upper_bound = mid;
            }
            else
            {
                lower_bound = mid + 1;
            }
        }

        return lower_bound;
    }

    template <typename RunOffsetT>
    HIPCUB_DEVICE __forceinline__ void InitWithRunOffsets(ItemT (&run_values)[RUNS_PER_THREAD],
                                                          RunOffsetT (&run_offsets)[RUNS_PER_THREAD])
    {

        RunOffsetT thread_dst_offset = static_cast<RunOffsetT>(linear_tid) * static_cast<RunOffsetT>(RUNS_PER_THREAD);
        #pragma unroll
        for (int i = 0; i < RUNS_PER_THREAD; i++)
        {
            temp_storage.runs.run_values[thread_dst_offset] = run_values[i];
            temp_storage.runs.run_offsets[thread_dst_offset] = run_offsets[i];
            thread_dst_offset++;
        }


        CTA_SYNC();
    }

    template <typename RunLengthT, typename TotalDecodedSizeT>
    HIPCUB_DEVICE __forceinline__ void InitWithRunLengths(ItemT (&run_values)[RUNS_PER_THREAD],
                                                          RunLengthT (&run_lengths)[RUNS_PER_THREAD],
                                                          TotalDecodedSizeT &total_decoded_size)
    {

        DecodedOffsetT run_offsets[RUNS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < RUNS_PER_THREAD; i++)
        {
            run_offsets[i] = static_cast<DecodedOffsetT>(run_lengths[i]);
        }
        DecodedOffsetT decoded_size_aggregate;
        RunOffsetScanT(this->temp_storage.offset_scan).ExclusiveSum(run_offsets, run_offsets, decoded_size_aggregate);
        total_decoded_size = static_cast<TotalDecodedSizeT>(decoded_size_aggregate);


        CTA_SYNC();

        InitWithRunOffsets(run_values, run_offsets);
    }

public:
    
    template <typename RelativeOffsetT>
    HIPCUB_DEVICE __forceinline__ void RunLengthDecode(ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                                                       RelativeOffsetT (&item_offsets)[DECODED_ITEMS_PER_THREAD],
                                                       DecodedOffsetT from_decoded_offset = 0)
    {

        DecodedOffsetT thread_decoded_offset = from_decoded_offset + linear_tid * DECODED_ITEMS_PER_THREAD;




        RunOffsetT assigned_run =
            StaticUpperBound<BLOCK_RUNS>(temp_storage.runs.run_offsets, BLOCK_RUNS, thread_decoded_offset) -
            static_cast<RunOffsetT>(1U);

        DecodedOffsetT assigned_run_begin = temp_storage.runs.run_offsets[assigned_run];


        DecodedOffsetT assigned_run_end = (assigned_run == BLOCK_RUNS - 1)
                                              ? thread_decoded_offset + DECODED_ITEMS_PER_THREAD
                                              : temp_storage.runs.run_offsets[assigned_run + 1];

        ItemT val = temp_storage.runs.run_values[assigned_run];

        #pragma unroll
        for (DecodedOffsetT i = 0; i < DECODED_ITEMS_PER_THREAD; i++)
        {
            decoded_items[i] = val;
            item_offsets[i] = thread_decoded_offset - assigned_run_begin;
            if (thread_decoded_offset == assigned_run_end - 1)
            {


                assigned_run++;
                assigned_run_begin = temp_storage.runs.run_offsets[assigned_run];


                assigned_run_end = (assigned_run == BLOCK_RUNS - 1) ? thread_decoded_offset + DECODED_ITEMS_PER_THREAD
                                                                    : temp_storage.runs.run_offsets[assigned_run + 1];
                val = temp_storage.runs.run_values[assigned_run];
            }
            thread_decoded_offset++;
        }
    }

    
    HIPCUB_DEVICE __forceinline__ void RunLengthDecode(ItemT (&decoded_items)[DECODED_ITEMS_PER_THREAD],
                                                       DecodedOffsetT from_decoded_offset = 0)
    {
        DecodedOffsetT item_offsets[DECODED_ITEMS_PER_THREAD];
        RunLengthDecode(decoded_items, item_offsets, from_decoded_offset);
    }
};

END_HIPCUB_NAMESPACE

#endif
