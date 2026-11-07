



 #ifndef HIPCUB_ROCPRIM_BLOCK_RADIX_RANK_SORT_OPERATIONS_HPP_
 #define HIPCUB_ROCPRIM_BLOCK_RADIX_RANK_SORT_OPERATIONS_HPP_

#include <type_traits>

#include "../config.hpp"

 #include <cub/rocprim/config.hpp>
 #include <cub/rocprim/type_traits.hpp>
 #include <cub/rocprim/detail/various.hpp>

BEGIN_HIPCUB_NAMESPACE


template <bool IS_DESCENDING, typename KeyT>
struct RadixSortTwiddle
{
    typedef Traits<KeyT> TraitsT;
    typedef typename TraitsT::UnsignedBits UnsignedBits;
    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits In(UnsignedBits key)
    {
        key = TraitsT::TwiddleIn(key);
        if (IS_DESCENDING) key = ~key;
        return key;
    }
    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits Out(UnsignedBits key)
    {
        if (IS_DESCENDING) key = ~key;
        key = TraitsT::TwiddleOut(key);
        return key;
    }
    static HIPCUB_HOST_DEVICE __forceinline__ UnsignedBits DefaultKey()
    {
        return Out(~UnsignedBits(0));
    }
};


    template <typename KeyT>
    struct BaseDigitExtractor
    {
        typedef Traits<KeyT> TraitsT;
        typedef typename TraitsT::UnsignedBits UnsignedBits;

        enum
        {
            FLOAT_KEY = TraitsT::CATEGORY == FLOATING_POINT,
        };

        static __device__ __forceinline__ UnsignedBits ProcessFloatMinusZero(UnsignedBits key)
        {
            if (!FLOAT_KEY) {
                return key;
            } else {
                UnsignedBits TWIDDLED_MINUS_ZERO_BITS =
                    TraitsT::TwiddleIn(UnsignedBits(1) << UnsignedBits(8 * sizeof(UnsignedBits) - 1));
                UnsignedBits TWIDDLED_ZERO_BITS = TraitsT::TwiddleIn(0);
                return key == TWIDDLED_MINUS_ZERO_BITS ? TWIDDLED_ZERO_BITS : key;
            }
        }
    };


    template <typename KeyT>
    struct BFEDigitExtractor : BaseDigitExtractor<KeyT>
    {
        using typename BaseDigitExtractor<KeyT>::UnsignedBits;

        uint32_t bit_start, num_bits;
        explicit __device__ __forceinline__ BFEDigitExtractor(
            uint32_t bit_start = 0, uint32_t num_bits = 0)
            : bit_start(bit_start), num_bits(num_bits)
        { }

        __device__ __forceinline__ uint32_t Digit(UnsignedBits key)
        {
            return BFE(this->ProcessFloatMinusZero(key), bit_start, num_bits);
        }
    };


    template <typename KeyT>
    struct ShiftDigitExtractor : BaseDigitExtractor<KeyT>
    {
        using typename BaseDigitExtractor<KeyT>::UnsignedBits;

        uint32_t bit_start, mask;
        explicit __device__ __forceinline__ ShiftDigitExtractor(
            uint32_t bit_start = 0, uint32_t num_bits = 0)
            : bit_start(bit_start), mask((1 << num_bits) - 1)
        { }

        __device__ __forceinline__ uint32_t Digit(UnsignedBits key)
        {
            return uint32_t(this->ProcessFloatMinusZero(key) >> UnsignedBits(bit_start)) & mask;
        }
    };

END_HIPCUB_NAMESPACE

#endif
