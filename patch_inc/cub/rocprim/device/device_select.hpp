



















#ifndef ROCPRIM_DEVICE_DEVICE_SELECT_HPP_
#define ROCPRIM_DEVICE_DEVICE_SELECT_HPP_

#include <type_traits>
#include <iterator>

#include "../config.hpp"
#include "../detail/various.hpp"
#include "../detail/binary_op_wrappers.hpp"

#include "../iterator/transform_iterator.hpp"

#include "device_scan.hpp"
#include "device_partition.hpp"

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
    class FlagIterator,
    class OutputIterator,
    class SelectedCountOutputIterator
>
inline
cudaError_t select(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  FlagIterator flags,
                  OutputIterator output,
                  SelectedCountOutputIterator selected_count_output,
                  const size_t size,
                  const cudaStream_t stream = 0,
                  const bool debug_synchronous = false)
{

    using unary_predicate_type = ::rocprim::empty_type;

    using inequality_op_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    rocprim::empty_type* const no_values = nullptr;

    return detail::partition_impl<detail::select_method::flag, true, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous, unary_predicate_type()
    );
}



















































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class SelectedCountOutputIterator,
    class UnaryPredicate
>
inline
cudaError_t select(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  SelectedCountOutputIterator selected_count_output,
                  const size_t size,
                  UnaryPredicate predicate,
                  const cudaStream_t stream = 0,
                  const bool debug_synchronous = false)
{

    using flag_type = ::rocprim::empty_type;
    using offset_type = unsigned int;
    flag_type * flags = nullptr;

    using inequality_op_type = ::rocprim::empty_type;
    rocprim::empty_type* const no_values = nullptr;

    return detail::partition_impl<detail::select_method::predicate, true, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, selected_count_output,
        size, inequality_op_type(), stream, debug_synchronous, predicate
    );
}











































































template<
    class Config = default_config,
    class InputIterator,
    class OutputIterator,
    class UniqueCountOutputIterator,
    class EqualityOp = ::rocprim::equal_to<typename std::iterator_traits<InputIterator>::value_type>
>
inline
cudaError_t unique(void * temporary_storage,
                  size_t& storage_size,
                  InputIterator input,
                  OutputIterator output,
                  UniqueCountOutputIterator unique_count_output,
                  const size_t size,
                  EqualityOp equality_op = EqualityOp(),
                  const cudaStream_t stream = 0,
                  const bool debug_synchronous = false)
{

    using unary_predicate_type = ::rocprim::empty_type;
    using offset_type = unsigned int;

    using flag_type = ::rocprim::empty_type;
    const flag_type * flags = nullptr;
    rocprim::empty_type* const no_values = nullptr;


    auto inequality_op = detail::inequality_wrapper<EqualityOp>(equality_op);

    return detail::partition_impl<detail::select_method::unique, true, Config, offset_type>(
        temporary_storage, storage_size, input, no_values, flags, output, no_values, unique_count_output,
        size, inequality_op, stream, debug_synchronous, unary_predicate_type()
    );
}














































template <typename Config = default_config,
          typename KeyIterator,
          typename ValueIterator,
          typename OutputKeyIterator,
          typename OutputValueIterator,
          typename UniqueCountOutputIterator,
          typename EqualityOp
          = ::rocprim::equal_to<typename std::iterator_traits<KeyIterator>::value_type>>
inline cudaError_t unique_by_key(void*                           temporary_storage,
                                size_t&                         storage_size,
                                const KeyIterator               keys_input,
                                const ValueIterator             values_input,
                                const OutputKeyIterator         keys_output,
                                const OutputValueIterator       values_output,
                                const UniqueCountOutputIterator unique_count_output,
                                const size_t                    size,
                                const EqualityOp                equality_op       = EqualityOp(),
                                const cudaStream_t               stream            = 0,
                                const bool                      debug_synchronous = false)
{
    using offset_type = unsigned int;

    ::rocprim::empty_type* const no_flags = nullptr;

    const auto no_predicate = ::rocprim::empty_type{};


    const auto inequality_op = detail::inequality_wrapper<EqualityOp>(equality_op);

    return detail::partition_impl<detail::select_method::unique, true, Config, offset_type>(
        temporary_storage,
        storage_size,
        keys_input,
        values_input,
        no_flags,
        keys_output,
        values_output,
        unique_count_output,
        size,
        inequality_op,
        stream,
        debug_synchronous,
        no_predicate);
}

#undef ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR




END_ROCPRIM_NAMESPACE

#endif
