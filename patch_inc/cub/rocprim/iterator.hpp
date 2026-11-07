



















#ifndef ROCPRIM_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_HPP_


#include "config.hpp"

#include "iterator/arg_index_iterator.hpp"
#include "iterator/constant_iterator.hpp"
#include "iterator/counting_iterator.hpp"
#include "iterator/discard_iterator.hpp"
#ifndef __HIP_CPU_RT__
#include "iterator/texture_cache_iterator.hpp"
#endif
#include "iterator/transform_iterator.hpp"
#include "iterator/zip_iterator.hpp"

#endif
