



















#ifndef ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_
#define ROCPRIM_BLOCK_BLOCK_RADIX_SORT_HPP_

#include <type_traits>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/radix_sort.hpp"
#include "../warp/detail/warp_scan_crosslane.hpp"

#include "../intrinsics.hpp"
#include "../functional.hpp"
#include "../types.hpp"

#include "block_exchange.hpp"




BEGIN_ROCPRIM_NAMESPACE

namespace detail
{





template<
    unsigned int BlockSizeX,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_bit_plus_scan
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;

    static constexpr unsigned int warp_size =
        detail::get_min_warp_size(BlockSize, ::rocprim::device_warp_size());

    static constexpr unsigned int warps_no = (BlockSize + warp_size - 1) / warp_size;





    using warp_scan_prefix_type =
        ::rocprim::detail::warp_scan_crosslane<unsigned int, detail::next_power_of_two(warps_no)>;

public:

    struct storage_type_
    {
        unsigned int warp_prefixes[warps_no];



    };

    using storage_type = detail::raw_storage<storage_type_>;

    template<unsigned int ItemsPerThread>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exclusive_scan(const unsigned int (&input)[ItemsPerThread],
                        unsigned int (&output)[ItemsPerThread],
                        unsigned int& reduction,
                        storage_type& storage)
    {
        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();
        const unsigned int lane_id = ::rocprim::lane_id();
        const unsigned int warp_id = ::rocprim::warp_id(flat_id);
        storage_type_& storage_ = storage.get();

        unsigned int warp_reduction = ::rocprim::bit_count(::rocprim::ballot(input[0]));
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            warp_reduction += ::rocprim::bit_count(::rocprim::ballot(input[i]));
        }
        if(lane_id == 0)
        {
            storage_.warp_prefixes[warp_id] = warp_reduction;
        }
        ::rocprim::syncthreads();


        if(flat_id < warps_no)
        {
            unsigned int prefix = storage_.warp_prefixes[flat_id];
            warp_scan_prefix_type().inclusive_scan(prefix, prefix, ::rocprim::plus<unsigned int>());
            storage_.warp_prefixes[flat_id] = prefix;
        }
#ifdef __HIP_CPU_RT__
        else
        {

            empty_type empty;
            ::rocprim::detail::warp_scan_crosslane<empty_type, detail::next_power_of_two(warps_no)>().inclusive_scan(empty, empty, empty_binary_op{});
        }
#endif
        ::rocprim::syncthreads();


        unsigned int lane_prefix = 0;
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            lane_prefix = ::rocprim::masked_bit_count(::rocprim::ballot(input[i]), lane_prefix);
        }


        output[0] = warp_id == 0
            ? lane_prefix
            : lane_prefix + storage_.warp_prefixes[warp_id - 1];
        for(unsigned int i = 1; i < ItemsPerThread; i++)
        {
            output[i] = output[i - 1] + input[i - 1];
        }


        reduction = storage_.warp_prefixes[warps_no - 1];
    }
};

}















































template<
    class Key,
    unsigned int BlockSizeX,
    unsigned int ItemsPerThread,
    class Value = empty_type,
    unsigned int BlockSizeY = 1,
    unsigned int BlockSizeZ = 1
>
class block_radix_sort
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    static constexpr bool with_values = !std::is_same<Value, empty_type>::value;

    using bit_key_type = typename ::rocprim::detail::radix_key_codec<Key>::bit_key_type;
    using bit_block_scan = detail::block_bit_plus_scan<BlockSizeX, BlockSizeY, BlockSizeZ>;

    using bit_keys_exchange_type = ::rocprim::block_exchange<bit_key_type, BlockSizeX, ItemsPerThread, BlockSizeY, BlockSizeZ>;
    using values_exchange_type = ::rocprim::block_exchange<Value, BlockSizeX, ItemsPerThread, BlockSizeY, BlockSizeZ>;


    struct storage_type_
    {
        union
        {
            typename bit_keys_exchange_type::storage_type bit_keys_exchange;
            typename values_exchange_type::storage_type values_exchange;
        };
        typename block_radix_sort<Key,BlockSizeX,ItemsPerThread,Value,BlockSizeY,BlockSizeZ>::bit_block_scan::storage_type bit_block_scan;
    };

public:









    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using storage_type = detail::raw_storage<storage_type_>;
    #else
    using storage_type = storage_type_;
    #endif











































    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<false>(keys, values, storage, begin_bit, end_bit);
    }














    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(keys, storage, begin_bit, end_bit);
    }











































    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<true>(keys, values, storage, begin_bit, end_bit);
    }














    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc(keys, storage, begin_bit, end_bit);
    }



















































    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              storage_type& storage,
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<false>(keys, values, storage, begin_bit, end_bit);
    }


















    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort(Key (&keys)[ItemsPerThread],
              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
              unsigned int begin_bit = 0,
              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort(keys, values, storage, begin_bit, end_bit);
    }



















































    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<true>(keys, values, storage, begin_bit, end_bit);
    }


















    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc(Key (&keys)[ItemsPerThread],
                   typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                   unsigned int begin_bit = 0,
                   unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc(keys, values, storage, begin_bit, end_bit);
    }












































    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit);
    }















    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_to_striped(keys, storage, begin_bit, end_bit);
    }












































    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        empty_type values[ItemsPerThread];
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit);
    }















    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_to_striped(keys, storage, begin_bit, end_bit);
    }



















































    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         storage_type& storage,
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<false, true>(keys, values, storage, begin_bit, end_bit);
    }
















    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_to_striped(Key (&keys)[ItemsPerThread],
                         typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                         unsigned int begin_bit = 0,
                         unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_to_striped(keys, values, storage, begin_bit, end_bit);
    }



















































    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                              storage_type& storage,
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        sort_impl<true, true>(keys, values, storage, begin_bit, end_bit);
    }















    template<bool WithValues = with_values>
    ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE
    void sort_desc_to_striped(Key (&keys)[ItemsPerThread],
                              typename std::enable_if<WithValues, Value>::type (&values)[ItemsPerThread],
                              unsigned int begin_bit = 0,
                              unsigned int end_bit = 8 * sizeof(Key))
    {
        ROCPRIM_SHARED_MEMORY storage_type storage;
        sort_desc_to_striped(keys, values, storage, begin_bit, end_bit);
    }

private:

    template<bool Descending, bool ToStriped = false, class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void sort_impl(Key (&keys)[ItemsPerThread],
                   SortedValue (&values)[ItemsPerThread],
                   storage_type& storage,
                   unsigned int begin_bit,
                   unsigned int end_bit)
    {
        using key_codec = ::rocprim::detail::radix_key_codec<Key, Descending>;
        storage_type_& storage_ = storage.get();

        const unsigned int flat_id = ::rocprim::flat_block_thread_id<BlockSizeX, BlockSizeY, BlockSizeZ>();

        bit_key_type bit_keys[ItemsPerThread];
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            bit_keys[i] = key_codec::encode(keys[i]);
        }


        for(unsigned int bit = begin_bit; bit < end_bit; bit++)
        {
            unsigned int bits[ItemsPerThread];
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {
                bits[i] = key_codec::extract_digit(bit_keys[i], bit, 1);
            }

            unsigned int ranks[ItemsPerThread];
#ifdef __HIP_CPU_RT__


            std::memset(ranks, 0, ItemsPerThread * sizeof(decltype(ranks[0])));
#endif
            unsigned int count;
            bit_block_scan().exclusive_scan(bits, ranks, count, storage_.bit_block_scan);


            const unsigned int start = BlockSize * ItemsPerThread - count;
            for(unsigned int i = 0; i < ItemsPerThread; i++)
            {

                ranks[i] = bits[i] != 0
                    ? (start + ranks[i])
                    : (flat_id * ItemsPerThread + i - ranks[i]);
            }
            exchange_keys(storage, bit_keys, ranks);
            exchange_values(storage, values, ranks);
        }

        if(ToStriped)
        {
            to_striped_keys(storage, bit_keys);
            to_striped_values(storage, values);
        }

        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            keys[i] = key_codec::decode(bit_keys[i]);
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_keys(storage_type& storage,
                       bit_key_type (&bit_keys)[ItemsPerThread],
                       const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();

        bit_keys_exchange_type().scatter_to_blocked(bit_keys, bit_keys, ranks, storage_.bit_keys_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_values(storage_type& storage,
                         SortedValue (&values)[ItemsPerThread],
                         const unsigned int (&ranks)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads();
        values_exchange_type().scatter_to_blocked(values, values, ranks, storage_.values_exchange);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void exchange_values(storage_type& storage,
                         empty_type (&values)[ItemsPerThread],
                         const unsigned int (&ranks)[ItemsPerThread])
    {
        (void) storage;
        (void) values;
        (void) ranks;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_striped_keys(storage_type& storage,
                         bit_key_type (&bit_keys)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads();
        bit_keys_exchange_type().blocked_to_striped(bit_keys, bit_keys, storage_.bit_keys_exchange);
    }

    template<class SortedValue>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_striped_values(storage_type& storage,
                           SortedValue (&values)[ItemsPerThread])
    {
        storage_type_& storage_ = storage.get();
        ::rocprim::syncthreads();
        values_exchange_type().blocked_to_striped(values, values, storage_.values_exchange);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void to_striped_values(storage_type& storage,
                           empty_type * values)
    {
        (void) storage;
        (void) values;
    }
};

END_ROCPRIM_NAMESPACE




#endif
