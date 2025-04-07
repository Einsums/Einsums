#ifndef _TBLIS_INTERNAL_3M_MULT_HPP_
#define _TBLIS_INTERNAL_3M_MULT_HPP_

#include "util/basic_types.h"
#include "util/thread.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_D, const T* D, stride_type inc_D,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);

}
}

#endif
