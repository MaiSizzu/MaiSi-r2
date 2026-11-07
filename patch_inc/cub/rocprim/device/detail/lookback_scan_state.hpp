



















#ifndef ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_
#define ROCPRIM_DEVICE_DETAIL_LOOKBACK_SCAN_STATE_HPP_

#include <type_traits>

#include "../../intrinsics.hpp"
#include "../../types.hpp"
#include "../../type_traits.hpp"

#include "../../warp/detail/warp_reduce_crosslane.hpp"
#include "../../warp/detail/warp_scan_crosslane.hpp"

#include "../../detail/various.hpp"
#include "../../detail/binary_op_wrappers.hpp"

extern "C"
{
    void __builtin_amdgcn_s_sleep(int);
}
BEGIN_ROCPRIM_NAMESPACE





namespace detail
{

enum prefix_flag
{

    PREFIX_INVALID = -1,

    PREFIX_EMPTY = 0,

    PREFIX_PARTIAL = 1,

    PREFIX_COMPLETE = 2
};





template<class T, bool UseSleep = false, bool IsSmall = (sizeof(T) <= 4)>
struct lookback_scan_state;


template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, true>
{
private:
    using flag_type_ = char;



    using prefix_underlying_type =
        typename std::conditional<
            (sizeof(T) > 2),
            unsigned long long,
            unsigned int
        >::type;


    struct alignas(sizeof(prefix_underlying_type)) prefix_type
    {
        flag_type_ flag;
        T value;
    };

    static_assert(sizeof(prefix_underlying_type) == sizeof(prefix_type), "");

public:

    using flag_type = flag_type_;
    using value_type = T;


    ROCPRIM_HOST static inline
    lookback_scan_state create(void* temp_storage, const unsigned int number_of_blocks)
    {
        (void) number_of_blocks;
        lookback_scan_state state;
        state.prefixes = reinterpret_cast<prefix_underlying_type*>(temp_storage);
        return state;
    }

    ROCPRIM_HOST static inline
    size_t get_storage_size(const unsigned int number_of_blocks)
    {
        return sizeof(prefix_underlying_type) * (::rocprim::host_warp_size() + number_of_blocks);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void initialize_prefix(const unsigned int block_id,
                           const unsigned int number_of_blocks)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        if(block_id < number_of_blocks)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_EMPTY;
            prefix_underlying_type p;
#ifndef __HIP_CPU_RT__
            __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
#else
            std::memcpy(&p, &prefix, sizeof(prefix_type));
#endif
            prefixes[padding + block_id] = p;
        }
        if(block_id < padding)
        {
            prefix_type prefix;
            prefix.flag = PREFIX_INVALID;
            prefix_underlying_type p;
#ifndef __HIP_CPU_RT__
            __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
#else
            std::memcpy(&p, &prefix, sizeof(prefix_type));
#endif
            prefixes[block_id] = p;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_partial(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_PARTIAL, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_complete(const unsigned int block_id, const T value)
    {
        this->set(block_id, PREFIX_COMPLETE, value);
    }


    ROCPRIM_DEVICE ROCPRIM_INLINE
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefix_type prefix;

        const unsigned int SLEEP_MAX = 32;
        unsigned int times_through = 1;

        prefix_underlying_type p = ::rocprim::detail::atomic_add(&prefixes[padding + block_id], 0);
#ifndef __HIP_CPU_RT__
        __builtin_memcpy(&prefix, &p, sizeof(prefix_type));
#else
        std::memcpy(&prefix, &p, sizeof(prefix_type));
#endif
        while(prefix.flag == PREFIX_EMPTY)
        {
            if (UseSleep)
            {
                for (unsigned int j = 0; j < times_through; j++)
#ifndef __HIP_CPU_RT__
                    __builtin_amdgcn_s_sleep(1);
#else
                    std::this_thread::sleep_for(std::chrono::microseconds{1});
#endif
                if (times_through < SLEEP_MAX)
                    times_through++;
            }

            prefix_underlying_type p = ::rocprim::detail::atomic_add(&prefixes[padding + block_id], 0);
#ifndef __HIP_CPU_RT__
            __builtin_memcpy(&prefix, &p, sizeof(prefix_type));
#else
            std::memcpy(&prefix, &p, sizeof(prefix_type));
#endif
        }


        flag = prefix.flag;
        value = prefix.value;
    }

private:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set(const unsigned int block_id, const flag_type flag, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        prefix_type prefix = { flag, value };
        prefix_underlying_type p;
#ifndef __HIP_CPU_RT__
        __builtin_memcpy(&p, &prefix, sizeof(prefix_type));
#else
        std::memcpy(&p, &prefix, sizeof(prefix_type));
#endif
        ::rocprim::detail::atomic_exch(&prefixes[padding + block_id], p);
    }

    prefix_underlying_type * prefixes;
};



template<class T, bool UseSleep>
struct lookback_scan_state<T, UseSleep, false>
{

public:
    using flag_type = char;
    using value_type = T;


    ROCPRIM_HOST static inline
    lookback_scan_state create(void* temp_storage, const unsigned int number_of_blocks)
    {
        const auto n = ::rocprim::host_warp_size() + number_of_blocks;
        lookback_scan_state state;

        auto ptr = static_cast<char*>(temp_storage);

        state.prefixes_flags = reinterpret_cast<flag_type*>(ptr);
        ptr += ::rocprim::detail::align_size(n * sizeof(flag_type));

        state.prefixes_partial_values = reinterpret_cast<T*>(ptr);
        ptr += ::rocprim::detail::align_size(n * sizeof(T));

        state.prefixes_complete_values = reinterpret_cast<T*>(ptr);
        return state;
    }

    ROCPRIM_HOST static inline
    size_t get_storage_size(const unsigned int number_of_blocks)
    {
        const auto n = ::rocprim::host_warp_size() + number_of_blocks;
        size_t size = ::rocprim::detail::align_size(n * sizeof(flag_type));
        size += 2 * ::rocprim::detail::align_size(n * sizeof(T));
        return size;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void initialize_prefix(const unsigned int block_id,
                           const unsigned int number_of_blocks)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();
        if(block_id < number_of_blocks)
        {
            prefixes_flags[padding + block_id] = PREFIX_EMPTY;
        }
        if(block_id < padding)
        {
            prefixes_flags[block_id] = PREFIX_INVALID;
        }
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_partial(const unsigned int block_id, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        store_volatile(&prefixes_partial_values[padding + block_id], value);
        ::rocprim::detail::memory_fence_device();
        store_volatile<flag_type>(&prefixes_flags[padding + block_id], PREFIX_PARTIAL);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void set_complete(const unsigned int block_id, const T value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        store_volatile(&prefixes_complete_values[padding + block_id], value);
        ::rocprim::detail::memory_fence_device();
        store_volatile<flag_type>(&prefixes_flags[padding + block_id], PREFIX_COMPLETE);
    }


    ROCPRIM_DEVICE ROCPRIM_INLINE
    void get(const unsigned int block_id, flag_type& flag, T& value)
    {
        constexpr unsigned int padding = ::rocprim::device_warp_size();

        const unsigned int SLEEP_MAX = 32;
        unsigned int times_through = 1;

        flag = load_volatile(&prefixes_flags[padding + block_id]);
        ::rocprim::detail::memory_fence_device();
        while(flag == PREFIX_EMPTY)
        {
            if (UseSleep)
            {
                for (unsigned int j = 0; j < times_through; j++)
#ifndef __HIP_CPU_RT__
                    __builtin_amdgcn_s_sleep(1);
#else
                    std::this_thread::sleep_for(std::chrono::microseconds{1});
#endif
                if (times_through < SLEEP_MAX)
                    times_through++;
            }

            flag = load_volatile(&prefixes_flags[padding + block_id]);
            ::rocprim::detail::memory_fence_device();
        }

        if(flag == PREFIX_PARTIAL)
            value = load_volatile(&prefixes_partial_values[padding + block_id]);
        else
            value = load_volatile(&prefixes_complete_values[padding + block_id]);
    }

private:
    flag_type * prefixes_flags;



    T * prefixes_partial_values;
    T * prefixes_complete_values;
};

template<class T, class BinaryFunction, class LookbackScanState>
class lookback_scan_prefix_op
{
    using flag_type = typename LookbackScanState::flag_type;
    static_assert(
        std::is_same<T, typename LookbackScanState::value_type>::value,
        "T must be LookbackScanState::value_type"
    );

public:
    ROCPRIM_DEVICE ROCPRIM_INLINE
    lookback_scan_prefix_op(unsigned int block_id,
                            BinaryFunction scan_op,
                            LookbackScanState &scan_state)
        : block_id_(block_id),
          scan_op_(scan_op),
          scan_state_(scan_state)
    {
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    ~lookback_scan_prefix_op() = default;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reduce_partial_prefixes(unsigned int block_id,
                                 flag_type& flag,
                                 T& partial_prefix)
    {



        using headflag_scan_op_type = reverse_binary_op_wrapper<
            BinaryFunction, T, T
        >;
        using warp_reduce_prefix_type = warp_reduce_crosslane<
            T, ::rocprim::device_warp_size(), false
        >;

        T block_prefix;
        scan_state_.get(block_id, flag, block_prefix);

        auto headflag_scan_op = headflag_scan_op_type(scan_op_);
        warp_reduce_prefix_type()
            .tail_segmented_reduce(
                block_prefix,
                partial_prefix,
                (flag == PREFIX_COMPLETE),
                headflag_scan_op
            );
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T get_prefix()
    {
        flag_type flag;
        T partial_prefix;
        unsigned int previous_block_id = block_id_ - ::rocprim::lane_id() - 1;



        reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
        T prefix = partial_prefix;


        while(::rocprim::detail::warp_all(flag != PREFIX_COMPLETE))
        {
            previous_block_id -= ::rocprim::device_warp_size();
            reduce_partial_prefixes(previous_block_id, flag, partial_prefix);
            prefix = scan_op_(partial_prefix, prefix);
        }
        return prefix;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    T operator()(T reduction)
    {

        if(::rocprim::lane_id() == 0)
        {
            scan_state_.set_partial(block_id_, reduction);
        }


        auto prefix = get_prefix();


        if(::rocprim::lane_id() == 0)
        {
            scan_state_.set_complete(block_id_, scan_op_(prefix, reduction));
        }
        return prefix;
    }

protected:
    unsigned int       block_id_;
    BinaryFunction     scan_op_;
    LookbackScanState& scan_state_;
};

inline cudaError_t is_sleep_scan_state_used(bool& use_sleep)
{
    cudaDeviceProp prop;
    int             deviceId;
    if(const cudaError_t error = cudaGetDevice(&deviceId))
    {
        return error;
    }
    else if(const cudaError_t error = cudaGetDeviceProperties(&prop, deviceId))
    {
        return error;
    }

    const int asicRevision = 0;
    use_sleep = 0;
    return cudaSuccess;
}

}

END_ROCPRIM_NAMESPACE

#endif
