#ifndef _TBLIS_IFACE_1T_ADD_H_
#define _TBLIS_IFACE_1T_ADD_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A,
                            tblis_tensor* B, const label_type* idx_B);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void add(T alpha, marray_view<const T> A, const label_type* idx_A,
         T  beta,       marray_view<T> B, const label_type* idx_B)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(beta, B);

    tblis_tensor_add(nullptr, nullptr, &A_s, idx_A, &B_s, idx_B);
}

template <typename T>
void add(const communicator& comm,
         T alpha, marray_view<const T> A, const label_type* idx_A,
         T  beta,       marray_view<T> B, const label_type* idx_B)
{
    tblis_tensor A_s(alpha, A);
    tblis_tensor B_s(beta, B);

    tblis_tensor_add(comm, nullptr, &A_s, idx_A, &B_s, idx_B);
}

template <typename T>
void add(const communicator& comm,
         T alpha, dpd_marray_view<const T> A, const label_type* idx_A,
         T  beta, dpd_marray_view<      T> B, const label_type* idx_B);

template <typename T>
void add(T alpha, dpd_marray_view<const T> A, const label_type* idx_A,
         T  beta, dpd_marray_view<      T> B, const label_type* idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_marray_view<const T> A, const label_type* idx_A,
         T  beta, indexed_marray_view<      T> B, const label_type* idx_B);

template <typename T>
void add(T alpha, indexed_marray_view<const T> A, const label_type* idx_A,
         T  beta, indexed_marray_view<      T> B, const label_type* idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_dpd_marray_view<const T> A, const label_type* idx_A,
         T  beta, indexed_dpd_marray_view<      T> B, const label_type* idx_B);

template <typename T>
void add(T alpha, indexed_dpd_marray_view<const T> A, const label_type* idx_A,
         T  beta, indexed_dpd_marray_view<      T> B, const label_type* idx_B)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            add(comm, alpha, A, idx_A, beta, B, idx_B);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef __cplusplus
}
#endif

#endif
