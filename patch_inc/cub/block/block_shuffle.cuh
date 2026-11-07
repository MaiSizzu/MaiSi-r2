

#ifndef HIPCUB_ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_
#define HIPCUB_ROCPRIM_BLOCK_BLOCK_SHUFFLE_HPP_

#include <type_traits>

#include "../config.hpp"

#include "../thread/thread_operators.cuh"

#include <cub/rocprim/block/block_shuffle.hpp>

BEGIN_HIPCUB_NAMESPACE



template <
    typename            T,
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y         = 1,
    int                 BLOCK_DIM_Z         = 1,
    int                 ARCH            = HIPCUB_ARCH>
class BlockShuffle : public ::rocprim::block_shuffle<
                    T,
                    BLOCK_DIM_X,
                    BLOCK_DIM_Y,
                    BLOCK_DIM_Z>
{
  static_assert(
      BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z > 0,
      "BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z must be greater than 0"
  );

  using base_type =
      typename ::rocprim::block_shuffle<
          T,
          BLOCK_DIM_X,
          BLOCK_DIM_Y,
          BLOCK_DIM_Z
      >;


  typename base_type::storage_type& temp_storage_;

public:
  using TempStorage = typename base_type::storage_type;

  HIPCUB_DEVICE inline
  BlockShuffle()  :      temp_storage_(private_storage())
  {}


  HIPCUB_DEVICE inline
  BlockShuffle(TempStorage &temp_storage)
  :      temp_storage_(temp_storage)
  {}

  
  HIPCUB_DEVICE inline void Offset(
      T   input,
      T&  output,
      int distance = 1)
  {
    base_type::offset(input,output,distance);
  }

  
  HIPCUB_DEVICE inline void Rotate(
      T   input,
      T&  output,
      unsigned int distance = 1)
  {
    base_type::rotate(input,output,distance);
  }
  
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Up(
    T (&input)[ITEMS_PER_THREAD],
    T (&prev)[ITEMS_PER_THREAD])
  {
    base_type::up(input,prev);
  }


   
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Up(
      T (&input)[ITEMS_PER_THREAD],
      T (&prev)[ITEMS_PER_THREAD],
      T &block_suffix)
  {
    base_type::up(input,prev,block_suffix);
  }

   
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Down(
      T (&input)[ITEMS_PER_THREAD],
      T (&next)[ITEMS_PER_THREAD])
  {
    base_type::down(input,next);
  }

   
  template <int ITEMS_PER_THREAD>
  HIPCUB_DEVICE inline void Down(
      T (&input)[ITEMS_PER_THREAD],
      T (&next)[ITEMS_PER_THREAD],
      T &block_prefix)
  {
    base_type::down(input,next,block_prefix);
  }

private:
    HIPCUB_DEVICE inline
    TempStorage& private_storage()
    {
        HIPCUB_SHARED_MEMORY TempStorage private_storage;
        return private_storage;
    }
};

END_HIPCUB_NAMESPACE

#endif
