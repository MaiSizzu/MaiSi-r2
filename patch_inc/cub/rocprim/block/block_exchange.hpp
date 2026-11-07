



















#ifndef ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_
#define ROCPRIM_BLOCK_BLOCK_EXCHANGE_HPP_

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"




BEGIN_ROCPRIM_NAMESPACE







































template<
    class T,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_exchange
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());

    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;




    static constexpr bool has_bank_conflicts =
        ItemsPerThread >= 2 && ::rocprim::detail::is_power_of_two(ItemsPerThread);
    static constexpr unsigned int banks_no = ::rocprim::detail::get_lds_banks_no();
    static constexpr unsigned int bank_conflicts_padding =
        has_bank_conflicts ? (BlockSize * ItemsPerThread / banks_no) : 0;


    struct storage_type_
    {
        T buffer[BlockSize * ItemsPerThread + bank_conflicts_padding];
    };

public:









    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_;
    #endif








    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        blocked_to_striped(input, output, storage);
    }






























    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(flat_id * ItemsPerThread + i)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(i * BlockSize + flat_id)];
        }
    }








    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        striped_to_blocked(input, output, storage);
    }






























    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void striped_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(i * BlockSize + flat_id)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }








    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        blocked_to_warp_striped(input, output, storage);
    }






























    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void blocked_to_warp_striped(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage)
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int offset = warp_id * items_per_warp;
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(offset + lane_id * ItemsPerThread + i)] = input[i];
        }

        ::rocprim::wave_barrier();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(offset + i * current_warp_size + lane_id)];
        }
    }








    template<class U>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        warp_striped_to_blocked(input, output, storage);
    }






























    template<class U>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void warp_striped_to_blocked(const T (&input)[ItemsPerThread],
                                 U (&output)[ItemsPerThread],
                                 storage_type& storage)
    {
        constexpr unsigned int items_per_warp = warp_size * ItemsPerThread;
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const unsigned int current_warp_size = get_current_warp_size();
        const unsigned int offset = warp_id * items_per_warp;
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(offset + i * current_warp_size + lane_id)] = input[i];
        }

        ::rocprim::wave_barrier();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(offset + lane_id * ItemsPerThread + i)];
        }
    }










    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_blocked(input, output, ranks, storage);
    }

    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void gather_from_striped(const T (&input)[ItemsPerThread],
                                   U (&output)[ItemsPerThread],
                                   const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        gather_from_striped(input, output, ranks, storage);
    }

































    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_blocked(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage_.buffer[index(rank)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[index(flat_id * ItemsPerThread + i)];
        }
    }

    template <class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void gather_from_striped(const T (&input)[ItemsPerThread],
                             U (&output)[ItemsPerThread],
                             const Offset (&ranks)[ItemsPerThread],
                             storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            storage_.buffer[index(i * BlockSize + flat_id)] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            output[i] = storage_.buffer[index(rank)];
        }
    }










    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_striped(input, output, ranks, storage);
    }

































    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped(const T (&input)[ItemsPerThread],
                            U (&output)[ItemsPerThread],
                            const Offset (&ranks)[ItemsPerThread],
                            storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            storage_.buffer[rank] = input[i];
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * BlockSize + flat_id];
        }
    }













    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_striped_guarded(input, output, ranks, storage);
    }




































    template<class U, class Offset>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped_guarded(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(rank >= 0)
            {
                storage_.buffer[rank] = input[i];
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * BlockSize + flat_id];
        }
    }












    template<class U, class Offset, class ValidFlag>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread])
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        scatter_to_striped_flagged(input, output, ranks, is_valid, storage);
    }





































    template<class U, class Offset, class ValidFlag>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void scatter_to_striped_flagged(const T (&input)[ItemsPerThread],
                                    U (&output)[ItemsPerThread],
                                    const Offset (&ranks)[ItemsPerThread],
                                    const ValidFlag (&is_valid)[ItemsPerThread],
                                    storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        storage_type_& storage_ = storage.get();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            const Offset rank = ranks[i];
            if(is_valid[i])
            {
                storage_.buffer[rank] = input[i];
            }
        }
        ::rocprim::syncthreads();

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output[i] = storage_.buffer[i * BlockSize + flat_id];
        }
    }

private:

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int get_current_warp_size() const
    {
        const unsigned int warp_id = ::rocprim::warp_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        return (warp_id == warps_no - 1)
            ? (BlockSize % warp_size > 0 ? BlockSize % warp_size : warp_size)
            : warp_size;
    }


    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int index(unsigned int n)
    {

        return has_bank_conflicts ? (n + n / banks_no) : n;
    }
};

END_ROCPRIM_NAMESPACE




#endif
