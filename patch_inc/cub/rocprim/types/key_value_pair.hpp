



















#ifndef ROCPRIM_TYPES_KEY_VALUE_PAIR_HPP_
#define ROCPRIM_TYPES_KEY_VALUE_PAIR_HPP_

#include "../config.hpp"




BEGIN_ROCPRIM_NAMESPACE

template<
    class Key_,
    class Value_
>
struct key_value_pair
{
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    using Key = Key_;
    using Value = Value_;
    #endif

    using key_type = Key_;
    using value_type = Value_;

    key_type key;
    value_type value;

    ROCPRIM_HOST_DEVICE inline
    key_value_pair() = default;

    ROCPRIM_HOST_DEVICE inline
    ~key_value_pair() = default;

    ROCPRIM_HOST_DEVICE inline
    key_value_pair(const key_type key, const value_type value) : key(key), value(value)
    {
    }

    #if __hcc_major__ < 1 || __hcc_major__ == 1 && __hcc_minor__ < 2
    ROCPRIM_HOST_DEVICE inline
    key_value_pair& operator =(const key_value_pair& kvb)
    {
        key = kvb.key;
        value = kvb.value;
        return *this;
    }
    #endif

    ROCPRIM_HOST_DEVICE inline
    bool operator !=(const key_value_pair& kvb)
    {
        return (key != kvb.key) || (value != kvb.value);
    }
};

END_ROCPRIM_NAMESPACE




#endif
