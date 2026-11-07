




#pragma once




#include "version.cuh"



#if (defined(CUB_NS_PREFIX) || defined(CUB_NS_POSTFIX)) && !defined(CUB_NS_QUALIFIER)
#error CUB requires a definition of CUB_NS_QUALIFIER when CUB_NS_PREFIX/POSTFIX are defined.
#endif


#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#define CUB_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE
#endif


#ifdef CUB_WRAPPED_NAMESPACE
#define CUB_NS_PREFIX                                                       \
  namespace CUB_WRAPPED_NAMESPACE                                           \
  {

#define CUB_NS_POSTFIX }

#define CUB_NS_QUALIFIER ::CUB_WRAPPED_NAMESPACE::cub
#endif


#ifndef CUB_NS_PREFIX
#define CUB_NS_PREFIX
#endif


#ifndef CUB_NS_POSTFIX
#define CUB_NS_POSTFIX
#endif


#ifndef CUB_NS_QUALIFIER
#define CUB_NS_QUALIFIER ::cub
#endif


#define CUB_NAMESPACE_BEGIN                                                 \
  CUB_NS_PREFIX                                                             \
  namespace cub                                                             \
  {


#define CUB_NAMESPACE_END                                                   \
  }                                                  \
  CUB_NS_POSTFIX


CUB_NS_PREFIX


namespace cub
{
}

CUB_NS_POSTFIX

