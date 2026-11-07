

#ifndef ROCPRIM_THREAD_THREAD_SCAN_HPP_
#define ROCPRIM_THREAD_THREAD_SCAN_HPP_


#include "../config.hpp"
#include "../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

 

 











 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_exclusive(
     T                   inclusive,
     T                   exclusive,
     T                   *input,
     T                   *output,
     ScanOp              scan_op,
     Int2Type<LENGTH>    )
 {
     ROCPRIM_UNROLL
     for (int i = 0; i < LENGTH; ++i)
     {
         inclusive = scan_op(exclusive, input[i]);
         output[i] = exclusive;
         exclusive = inclusive;
     }

     return inclusive;
 }













 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_exclusive(
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

     return thread_scan_exclusive(inclusive, exclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }











 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_exclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     return thread_scan_exclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
 }










 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_inclusive(
     T                   inclusive,
     T                   *input,
     T                   *output,
     ScanOp              scan_op,
     Int2Type<LENGTH>    )
 {
     ROCPRIM_UNROLL
     for (int i = 0; i < LENGTH; ++i)
     {
         inclusive = scan_op(inclusive, input[i]);
         output[i] = inclusive;
     }

     return inclusive;
 }










 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_inclusive(
     T           *input,
     T           *output,
     ScanOp      scan_op)
 {
     T inclusive = input[0];
     output[0] = inclusive;


     return thread_scan_inclusive(inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }










 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_inclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op)
 {
     return thread_scan_inclusive<LENGTH>((T*) input, (T*) output, scan_op);
 }












 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_inclusive(
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


     return thread_scan_inclusive(inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }












 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE ROCPRIM_INLINE
 T thread_scan_inclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     return thread_scan_inclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
 }




 

 END_ROCPRIM_NAMESPACE

 #endif
