



















#ifndef ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_ATOMIC_HPP_
#define ROCPRIM_BLOCK_DETAIL_BLOCK_HISTOGRAM_ATOMIC_HPP_

#include <type_traits>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<
    class T,
    unsigned int BlockSizeX,
    unsigned int BlockSizeY,
    unsigned int BlockSizeZ,
    unsigned int ItemsPerThread,
    unsigned int Bins
>
class block_histogram_atomic
{
    static constexpr unsigned int BlockSize = BlockSizeX * BlockSizeY * BlockSizeZ;
    static_assert(
        std::is_convertible<T, unsigned int>::value,
        "T must be convertible to unsigned int"
    );

public:
    using storage_type = typename ::rocprim::detail::empty_storage_type;

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins])
    {
        static_assert(
            std::is_same<Counter, unsigned int>::value || std::is_same<Counter, int>::value ||
            std::is_same<Counter, float>::value || std::is_same<Counter, unsigned long long>::value,
            "Counter must be type that is supported by atomics (float, int, unsigned int, unsigned long long)"
        );
        ROCPRIM_UNROLL
        for (unsigned int i = 0; i < ItemsPerThread; ++i)
        {
              ::rocprim::detail::atomic_add(&hist[static_cast<unsigned int>(input[i])], Counter(1));
        }
        ::rocprim::syncthreads();
    }

    template<class Counter>
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void composite(T (&input)[ItemsPerThread],
                   Counter hist[Bins],
                   storage_type& storage)
    {
        (void) storage;
        this->composite(input, hist);
    }
};

}

END_ROCPRIM_NAMESPACE

#endif
