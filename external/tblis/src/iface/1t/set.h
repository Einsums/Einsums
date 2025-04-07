#ifndef _TBLIS_IFACE_1T_SET_H_
#define _TBLIS_IFACE_1T_SET_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_tensor* A, const label_type* idx_A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void set(T alpha, marray_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A);

    tblis_tensor_set(nullptr, nullptr, &alpha_s, &A_s, idx_A);
}

template <typename T>
void set(const communicator& comm, T alpha, marray_view<T> A, const label_type* idx_A)
{
    tblis_scalar alpha_s(alpha);
    tblis_tensor A_s(A);

    tblis_tensor_set(comm, nullptr, &alpha_s, &A_s, idx_A);
}

template <typename T>
void set(const communicator& comm,
         T alpha, dpd_marray_view<T> A, const label_type* idx_A);

template <typename T>
void set(T alpha, dpd_marray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void set(const communicator& comm,
         T alpha, indexed_marray_view<T> A, const label_type* idx_A);

template <typename T>
void set(T alpha, indexed_marray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

template <typename T>
void set(const communicator& comm,
         T alpha, indexed_dpd_marray_view<T> A, const label_type* idx_A);

template <typename T>
void set(T alpha, indexed_dpd_marray_view<T> A, const label_type* idx_A)
{
    parallelize
    (
        [&](const communicator& comm)
        {
            set(comm, alpha, A, idx_A);
        },
        tblis_get_num_threads()
    );
}

#endif

#ifdef __cplusplus
}
#endif

#endif
