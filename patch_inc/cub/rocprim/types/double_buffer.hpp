



















#ifndef ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_
#define ROCPRIM_TYPES_DOUBLE_BUFFER_HPP_

#include "../config.hpp"




BEGIN_ROCPRIM_NAMESPACE

template<class T>
class double_buffer
{
    T * buffers[2];

    unsigned int selector;

public:

    ROCPRIM_HOST_DEVICE inline
    double_buffer()
    {
        selector = 0;
        buffers[0] = nullptr;
        buffers[1] = nullptr;
    }

    ROCPRIM_HOST_DEVICE inline
    double_buffer(T * current, T * alternate)
    {
        selector = 0;
        buffers[0] = current;
        buffers[1] = alternate;
    }

    ROCPRIM_HOST_DEVICE inline
    T * current() const
    {
        return buffers[selector];
    }

    ROCPRIM_HOST_DEVICE inline
    T * alternate() const
    {
        return buffers[selector ^ 1];
    }

    ROCPRIM_HOST_DEVICE inline
    void swap()
    {
        selector ^= 1;
    }
};

END_ROCPRIM_NAMESPACE




#endif
