

#ifndef HIPCUB_ROCPRIM_UTIL_ALLOCATOR_HPP_
#define HIPCUB_ROCPRIM_UTIL_ALLOCATOR_HPP_

#include "config.hpp"

#include <set>
#include <map>
#include <mutex>

#include <math.h>
#include <stdio.h>

BEGIN_HIPCUB_NAMESPACE

#define _HipcubLog(format, ...) printf(format, __VA_ARGS__);



struct CachingDeviceAllocator
{





    static const unsigned int INVALID_BIN = (unsigned int) -1;


    static const size_t INVALID_SIZE = (size_t) -1;


    static const int INVALID_DEVICE_ORDINAL = -1;





    
    struct BlockDescriptor
    {
        void*           d_ptr;
        size_t          bytes;
        unsigned int    bin;
        int             device;
        cudaStream_t     associated_stream;
        cudaEvent_t      ready_event;


        BlockDescriptor(void *d_ptr, int device) :
            d_ptr(d_ptr),
            bytes(0),
            bin(INVALID_BIN),
            device(device),
            associated_stream(0),
            ready_event(0)
        {}


        BlockDescriptor(int device) :
            d_ptr(NULL),
            bytes(0),
            bin(INVALID_BIN),
            device(device),
            associated_stream(0),
            ready_event(0)
        {}


        static bool PtrCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device == b.device)
                return (a.d_ptr < b.d_ptr);
            else
                return (a.device < b.device);
        }


        static bool SizeCompare(const BlockDescriptor &a, const BlockDescriptor &b)
        {
            if (a.device == b.device)
                return (a.bytes < b.bytes);
            else
                return (a.device < b.device);
        }
    };


    typedef bool (*Compare)(const BlockDescriptor &, const BlockDescriptor &);

    class TotalBytes {
    public:
        size_t free;
        size_t live;
        TotalBytes() { free = live = 0; }
    };


    typedef std::multiset<BlockDescriptor, Compare> CachedBlocks;


    typedef std::multiset<BlockDescriptor, Compare> BusyBlocks;


    typedef std::map<int, TotalBytes> GpuCachedBytes;






    
    static unsigned int IntPow(
        unsigned int base,
        unsigned int exp)
    {
        unsigned int retval = 1;
        while (exp > 0)
        {
            if (exp & 1) {
                retval = retval * base;
            }
            base = base * base;
            exp = exp >> 1;
        }
        return retval;
    }


    
    void NearestPowerOf(
        unsigned int    &power,
        size_t          &rounded_bytes,
        unsigned int    base,
        size_t          value)
    {
        power = 0;
        rounded_bytes = 1;

        if (value * base < value)
        {

            power = sizeof(size_t) * 8;
            rounded_bytes = size_t(0) - 1;
            return;
        }

        while (rounded_bytes < value)
        {
            rounded_bytes *= base;
            power++;
        }
    }






    std::mutex      mutex;

    unsigned int    bin_growth;
    unsigned int    min_bin;
    unsigned int    max_bin;

    size_t          min_bin_bytes;
    size_t          max_bin_bytes;
    size_t          max_cached_bytes;

    const bool      skip_cleanup;
    bool            debug;

    GpuCachedBytes  cached_bytes;
    CachedBlocks    cached_blocks;
    BusyBlocks      live_blocks;





    
    CachingDeviceAllocator(
        unsigned int    bin_growth,
        unsigned int    min_bin             = 1,
        unsigned int    max_bin             = INVALID_BIN,
        size_t          max_cached_bytes    = INVALID_SIZE,
        bool            skip_cleanup        = false,
        bool            debug               = false)
    :
        bin_growth(bin_growth),
        min_bin(min_bin),
        max_bin(max_bin),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes(max_cached_bytes),
        skip_cleanup(skip_cleanup),
        debug(debug),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare)
    {}


    
    CachingDeviceAllocator(
        bool skip_cleanup = false,
        bool debug = false)
    :
        bin_growth(8),
        min_bin(3),
        max_bin(7),
        min_bin_bytes(IntPow(bin_growth, min_bin)),
        max_bin_bytes(IntPow(bin_growth, max_bin)),
        max_cached_bytes((max_bin_bytes * 3) - 1),
        skip_cleanup(skip_cleanup),
        debug(debug),
        cached_blocks(BlockDescriptor::SizeCompare),
        live_blocks(BlockDescriptor::PtrCompare)
    {}


    
    cudaError_t SetMaxCachedBytes(
        size_t max_cached_bytes)
    {

        mutex.lock();

        if (debug) _HipcubLog("Changing max_cached_bytes (%lld -> %lld)\n", (long long) this->max_cached_bytes, (long long) max_cached_bytes);

        this->max_cached_bytes = max_cached_bytes;


        mutex.unlock();

        return cudaSuccess;
    }


    
    cudaError_t DeviceAllocate(
        int             device,
        void            **d_ptr,
        size_t          bytes,
        cudaStream_t     active_stream = 0)
    {
        *d_ptr                          = NULL;
        int entrypoint_device           = INVALID_DEVICE_ORDINAL;
        cudaError_t error                = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL)
        {
            if (cubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
            device = entrypoint_device;
        }


        bool found = false;
        BlockDescriptor search_key(device);
        search_key.associated_stream = active_stream;
        NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

        if (search_key.bin > max_bin)
        {



            search_key.bin      = INVALID_BIN;
            search_key.bytes    = bytes;
        }
        else
        {

            mutex.lock();

            if (search_key.bin < min_bin)
            {

                search_key.bin      = min_bin;
                search_key.bytes    = min_bin_bytes;
            }


            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
            while ((block_itr != cached_blocks.end())
                    && (block_itr->device == device)
                    && (block_itr->bin == search_key.bin))
            {



                if ((active_stream == block_itr->associated_stream) ||
                    (cudaEventQuery(block_itr->ready_event) != cudaErrorNotReady))
                {

                    found = true;
                    search_key = *block_itr;
                    search_key.associated_stream = active_stream;
                    live_blocks.insert(search_key);


                    cached_bytes[device].free -= search_key.bytes;
                    cached_bytes[device].live += search_key.bytes;

                    if (debug) _HipcubLog("\tDevice %d reused cached block at %p (%lld bytes) for stream %lld (previously associated with stream %lld).\n",
                        device, search_key.d_ptr, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long)  block_itr->associated_stream);

                    cached_blocks.erase(block_itr);

                    break;
                }
                block_itr++;
            }


            mutex.unlock();
        }


        if (!found)
        {

            if (device != entrypoint_device)
            {
                if (cubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
                if (cubDebug(error = cudaSetDevice(device))) return error;
            }


            if (cubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes)) == cudaErrorMemoryAllocation)
            {

                if (debug) _HipcubLog("\tDevice %d failed to allocate %lld bytes for stream %lld, retrying after freeing cached allocations",
                      device, (long long) search_key.bytes, (long long) search_key.associated_stream);

                error = cudaGetLastError();
                error = cudaSuccess;


                mutex.lock();


                BlockDescriptor free_key(device);
                CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);

                while ((block_itr != cached_blocks.end()) && (block_itr->device == device))
                {





                    if (cubDebug(error = cudaFree(block_itr->d_ptr))) break;
                    if (cubDebug(error = cudaEventDestroy(block_itr->ready_event))) break;


                    cached_bytes[device].free -= block_itr->bytes;

                    if (debug) _HipcubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                        device, (long long) block_itr->bytes, (long long) cached_blocks.size(), (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].live);

                    cached_blocks.erase(block_itr);

                    block_itr++;
                }


                mutex.unlock();


                if (error) return error;


                if (cubDebug(error = cudaMalloc(&search_key.d_ptr, search_key.bytes))) return error;
            }


            if (cubDebug(error = cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming)))
                return error;


            mutex.lock();
            live_blocks.insert(search_key);
            cached_bytes[device].live += search_key.bytes;
            mutex.unlock();

            if (debug) _HipcubLog("\tDevice %d allocated new device block at %p (%lld bytes associated with stream %lld).\n",
                      device, search_key.d_ptr, (long long) search_key.bytes, (long long) search_key.associated_stream);


            if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
            {
                if (cubDebug(error = cudaSetDevice(entrypoint_device))) return error;
            }
        }


        *d_ptr = search_key.d_ptr;

        if (debug) _HipcubLog("\t\t%lld available blocks cached (%lld bytes), %lld live blocks outstanding(%lld bytes).\n",
            (long long) cached_blocks.size(), (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].live);

        return error;
    }


    
    cudaError_t DeviceAllocate(
        void            **d_ptr,
        size_t          bytes,
        cudaStream_t     active_stream = 0)
    {
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    }


    
    cudaError_t DeviceFree(
        int             device,
        void*           d_ptr)
    {
        int entrypoint_device           = INVALID_DEVICE_ORDINAL;
        cudaError_t error                = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL)
        {
            if (cubDebug(error = cudaGetDevice(&entrypoint_device)))
                return error;
            device = entrypoint_device;
        }


        mutex.lock();


        bool recached = false;
        BlockDescriptor search_key(d_ptr, device);
        BusyBlocks::iterator block_itr = live_blocks.find(search_key);
        if (block_itr != live_blocks.end())
        {

            search_key = *block_itr;
            live_blocks.erase(block_itr);
            cached_bytes[device].live -= search_key.bytes;


            if ((search_key.bin != INVALID_BIN) && (cached_bytes[device].free + search_key.bytes <= max_cached_bytes))
            {

                recached = true;
                cached_blocks.insert(search_key);
                cached_bytes[device].free += search_key.bytes;

                if (debug) _HipcubLog("\tDevice %d returned %lld bytes from associated stream %lld.\n\t\t %lld available blocks cached (%lld bytes), %lld live blocks outstanding. (%lld bytes)\n",
                    device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(),
                    (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].live);
            }
        }


        if (device != entrypoint_device)
        {
            if (cubDebug(error = cudaGetDevice(&entrypoint_device))) return error;
            if (cubDebug(error = cudaSetDevice(device))) return error;
        }

        if (recached)
        {

            if (cubDebug(error = cudaEventRecord(search_key.ready_event, search_key.associated_stream))) return error;
        }


        mutex.unlock();

        if (!recached)
        {

            if (cubDebug(error = cudaFree(d_ptr))) return error;
            if (cubDebug(error = cudaEventDestroy(search_key.ready_event))) return error;

            if (debug) _HipcubLog("\tDevice %d freed %lld bytes from associated stream %lld.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                device, (long long) search_key.bytes, (long long) search_key.associated_stream, (long long) cached_blocks.size(), (long long) cached_bytes[device].free, (long long) live_blocks.size(), (long long) cached_bytes[device].live);
        }


        if ((entrypoint_device != INVALID_DEVICE_ORDINAL) && (entrypoint_device != device))
        {
            if (cubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;
    }


    
    cudaError_t DeviceFree(
        void*           d_ptr)
    {
        return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
    }


    
    cudaError_t FreeAllCached()
    {
        cudaError_t error          = cudaSuccess;
        int entrypoint_device     = INVALID_DEVICE_ORDINAL;
        int current_device        = INVALID_DEVICE_ORDINAL;

        mutex.lock();

        while (!cached_blocks.empty())
        {

            CachedBlocks::iterator begin = cached_blocks.begin();


            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
            {
                if (cubDebug(error = cudaGetDevice(&entrypoint_device))) break;
            }


            if (begin->device != current_device)
            {
                if (cubDebug(error = cudaSetDevice(begin->device))) break;
                current_device = begin->device;
            }


            if (cubDebug(error = cudaFree(begin->d_ptr))) break;
            if (cubDebug(error = cudaEventDestroy(begin->ready_event))) break;


            cached_bytes[current_device].free -= begin->bytes;

            if (debug) _HipcubLog("\tDevice %d freed %lld bytes.\n\t\t  %lld available blocks cached (%lld bytes), %lld live blocks (%lld bytes) outstanding.\n",
                current_device, (long long) begin->bytes, (long long) cached_blocks.size(), (long long) cached_bytes[current_device].free, (long long) live_blocks.size(), (long long) cached_bytes[current_device].live);

            cached_blocks.erase(begin);
        }

        mutex.unlock();


        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
        {
            if (cubDebug(error = cudaSetDevice(entrypoint_device))) return error;
        }

        return error;
    }


    
    virtual ~CachingDeviceAllocator()
    {
        if (!skip_cleanup)
            FreeAllCached();
    }

};

END_HIPCUB_NAMESPACE

#endif
