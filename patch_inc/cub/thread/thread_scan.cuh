

#ifndef HIBCUB_ROCPRIM_THREAD_THREAD_SCAN_HPP_
#define HIBCUB_ROCPRIM_THREAD_THREAD_SCAN_HPP_


#include "../config.hpp"
#include "../util_type.cuh"

BEGIN_HIPCUB_NAMESPACE


namespace internal {

 

 

 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanExclusive(
     T                   inclusive,
     T                   exclusive,
     T                   *input,
     T                   *output,
     ScanOp              scan_op,
     Int2Type<LENGTH>    )
 {
     #pragma unroll
     for (int i = 0; i < LENGTH; ++i)
     {
         inclusive = scan_op(exclusive, input[i]);
         output[i] = exclusive;
         exclusive = inclusive;
     }

     return inclusive;
 }

 #ifndef DOXYGEN_SHOULD_SKIP_THIS

 
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanExclusive(
     T           *input,
     T           *output,
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     T inclusive = input[0];
     if (apply_prefix)
     {
         inclusive = scan_op(prefix, inclusive);
     }
     output[0] = prefix;
     T exclusive = inclusive;

     return ThreadScanExclusive(inclusive, exclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }

 
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanExclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     return ThreadScanExclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
 }

 #endif

 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanInclusive(
     T                   inclusive,
     T                   *input,
     T                   *output,
     ScanOp              scan_op,
     Int2Type<LENGTH>    )
 {
     #pragma unroll
     for (int i = 0; i < LENGTH; ++i)
     {
         inclusive = scan_op(inclusive, input[i]);
         output[i] = inclusive;
     }

     return inclusive;
 }

#ifndef DOXYGEN_SHOULD_SKIP_THIS

 
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanInclusive(
     T           *input,
     T           *output,
     ScanOp      scan_op)
 {
     T inclusive = input[0];
     output[0] = inclusive;


     return ThreadScanInclusive(inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }

 
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanInclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op)
 {
     return ThreadScanInclusive<LENGTH>((T*) input, (T*) output, scan_op);
 }

 
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanInclusive(
     T           *input,
     T           *output,
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     T inclusive = input[0];
     if (apply_prefix)
     {
         inclusive = scan_op(prefix, inclusive);
     }
     output[0] = inclusive;


     return ThreadScanInclusive(inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }

 
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 __device__ __forceinline__ T ThreadScanInclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     return ThreadScanInclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
 }

 #endif



 


 }

 END_HIPCUB_NAMESPACE

 #endif
