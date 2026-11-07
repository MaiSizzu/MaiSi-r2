



















#ifndef ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_
#define ROCPRIM_DEVICE_DETAIL_ORDERED_BLOCK_ID_HPP_

#include <type_traits>
#include <limits>

#include "../../detail/various.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{


template<class T  = unsigned int>
struct ordered_block_id
{
    static_assert(std::is_integral<T>::value, "T must be integer");
    using id_type = T;


    struct storage_type
    {
        id_type id;
    };

    ROCPRIM_HOST static inline
    ordered_block_id create(id_type * id)
    {
        ordered_block_id ordered_id;
        ordered_id.id = id;
        return ordered_id;
    }

    ROCPRIM_HOST static inline
    size_t get_storage_size()
    {
        return sizeof(id_type);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void reset()
    {
        *id = static_cast<id_type>(0);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    id_type get(unsigned int tid, storage_type& storage)
    {
        if(tid == 0)
        {
            storage.id = ::rocprim::detail::atomic_add(this->id, 1);
        }
        ::rocprim::syncthreads();
        return storage.id;
    }

    id_type* id;
};

}

END_ROCPRIM_NAMESPACE

#endif
