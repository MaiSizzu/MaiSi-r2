



















#ifndef ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"

#include "../iterator/constant_iterator.hpp"
#include "../iterator/counting_iterator.hpp"
#include "../iterator/discard_iterator.hpp"
#include "../iterator/zip_iterator.hpp"

#include "device_run_length_encode_config.hpp"
#include "device_reduce_by_key.hpp"
#include "device_select.hpp"

BEGIN_ROCPRIM_NAMESPACE




namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start) \
    { \
        if(error != cudaSuccess) return error; \
        if(debug_synchronous) \
        { \
            std::cout << name << "(" << size << ")"; \
            auto error = cudaStreamSynchronize(stream); \
            if(error != cudaSuccess) return error; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto d = std::chrono::duration_cast<std::chrono::duration<double>>(end - start); \
            std::cout << " " << d.count() * 1000 << " ms" << '\n'; \
        } \
    }

}

















































































template<
    class Config = default_config,
    class InputIterator,
    class UniqueOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
cudaError_t run_length_encode(void * temporary_storage,
                             size_t& storage_size,
                             InputIterator input,
                             unsigned int size,
                             UniqueOutputIterator unique_output,
                             CountsOutputIterator counts_output,
                             RunsCountOutputIterator runs_count_output,
                             cudaStream_t stream = 0,
                             bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using count_type = unsigned int;

    using config = detail::default_or_custom_config<
        Config,
        detail::default_run_length_encode_config
    >;

    return ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        temporary_storage, storage_size,
        input, make_constant_iterator<count_type>(1), size,
        unique_output, counts_output, runs_count_output,
        ::rocprim::plus<count_type>(), ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
}


















































































template<
    class Config = default_config,
    class InputIterator,
    class OffsetsOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
cudaError_t run_length_encode_non_trivial_runs(void * temporary_storage,
                                              size_t& storage_size,
                                              InputIterator input,
                                              unsigned int size,
                                              OffsetsOutputIterator offsets_output,
                                              CountsOutputIterator counts_output,
                                              RunsCountOutputIterator runs_count_output,
                                              cudaStream_t stream = 0,
                                              bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using offset_type = unsigned int;
    using count_type = unsigned int;
    using offset_count_pair = typename ::rocprim::tuple<offset_type, count_type>;

    using config = detail::default_or_custom_config<
        Config,
        detail::default_run_length_encode_config
    >;

    cudaError_t error;

    auto reduce_op = [] __device__ (const offset_count_pair& a, const offset_count_pair& b)
    {
        return offset_count_pair(
            ::rocprim::get<0>(a),
            ::rocprim::get<1>(a) + ::rocprim::get<1>(b)
        );
    };
    auto non_trivial_runs_select_op = [] __device__ (const offset_count_pair& a)
    {
        return ::rocprim::get<1>(a) > 1;
    };

    offset_type * offsets_tmp = nullptr;
    count_type * counts_tmp = nullptr;
    count_type * all_runs_count_tmp = nullptr;


    size_t reduce_by_key_bytes;
    error = ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        nullptr, reduce_by_key_bytes,
        input,
        ::rocprim::make_zip_iterator(
            ::rocprim::make_tuple(
                ::rocprim::make_counting_iterator<offset_type>(0),
                ::rocprim::make_constant_iterator<count_type>(1)
            )
        ),
        size,
        ::rocprim::make_discard_iterator(),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        all_runs_count_tmp,
        reduce_op, ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
    if(error != cudaSuccess) return error;
    reduce_by_key_bytes = ::rocprim::detail::align_size(reduce_by_key_bytes);


    size_t select_bytes;
    error = ::rocprim::select<typename config::select>(
        nullptr, select_bytes,
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_output, counts_output)),
        runs_count_output,
        size,
        non_trivial_runs_select_op,
        stream, debug_synchronous
    );
    if(error != cudaSuccess) return error;
    select_bytes = ::rocprim::detail::align_size(select_bytes);

    const size_t offsets_tmp_bytes = ::rocprim::detail::align_size(size * sizeof(offset_type));
    const size_t counts_tmp_bytes = ::rocprim::detail::align_size(size * sizeof(count_type));
    const size_t all_runs_count_tmp_bytes = sizeof(count_type);
    if(temporary_storage == nullptr)
    {
        storage_size = ::rocprim::max(reduce_by_key_bytes, select_bytes) +
            offsets_tmp_bytes + counts_tmp_bytes + all_runs_count_tmp_bytes;
        return cudaSuccess;
    }

    char * ptr = reinterpret_cast<char *>(temporary_storage);
    ptr += ::rocprim::max(reduce_by_key_bytes, select_bytes);
    offsets_tmp = reinterpret_cast<offset_type *>(ptr);
    ptr += offsets_tmp_bytes;
    counts_tmp = reinterpret_cast<count_type *>(ptr);
    ptr += counts_tmp_bytes;
    all_runs_count_tmp = reinterpret_cast<count_type *>(ptr);

    std::chrono::high_resolution_clock::time_point start;

    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    error = ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        temporary_storage, reduce_by_key_bytes,
        input,
        ::rocprim::make_zip_iterator(
            ::rocprim::make_tuple(
                ::rocprim::make_counting_iterator<offset_type>(0),
                ::rocprim::make_constant_iterator<count_type>(1)
            )
        ),
        size,
        ::rocprim::make_discard_iterator(),
        ::rocprim::make_zip_iterator(rocprim::make_tuple(offsets_tmp, counts_tmp)),
        all_runs_count_tmp,
        reduce_op, ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::reduce_by_key", size, start)


    count_type all_runs_count;

    error = cudaMemcpyAsync(&all_runs_count, all_runs_count_tmp, sizeof(count_type), cudaMemcpyDeviceToHost, stream);
    if(error != cudaSuccess) return error;
    error = cudaStreamSynchronize(stream);




    if(debug_synchronous) start = std::chrono::high_resolution_clock::now();
    error = ::rocprim::select<typename config::select>(
        temporary_storage, select_bytes,
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_tmp, counts_tmp)),
        ::rocprim::make_zip_iterator(::rocprim::make_tuple(offsets_output, counts_output)),
        runs_count_output,
        all_runs_count,
        non_trivial_runs_select_op,
        stream, debug_synchronous
    );
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("rocprim::select", all_runs_count, start)

    return cudaSuccess;
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR




END_ROCPRIM_NAMESPACE

#endif
