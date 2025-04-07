#ifndef _TBLIS_NODES_GEMM_UKR_HPP_
#define _TBLIS_NODES_GEMM_UKR_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "matrix/normal_matrix.hpp"
#include "matrix/block_scatter_matrix.hpp"
#include "matrix/patch_block_scatter_matrix.hpp"

#include "configs/configs.hpp"

namespace tblis
{

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c,
                 const stride_type* TBLIS_RESTRICT rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[rs_c[i] + j*cs_c];
            }
        }
    }
}

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c,
                 stride_type rs_c, const stride_type* TBLIS_RESTRICT cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[i*rs_c + cs_c[j]];
            }
        }
    }
}

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c,
                 const stride_type* TBLIS_RESTRICT rs_c,
                 const stride_type* TBLIS_RESTRICT cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[rs_c[i] + cs_c[j]];
            }
        }
    }
}

struct gemm_micro_kernel
{
    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, normal_matrix<T>& A,
                             normal_matrix<T>& B,
                    T  beta, normal_matrix<T>& C) const
    {
        (void)comm;

        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const bool flip_ukr = cfg.gemm_flip_ukr.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        const T* p_a = A.data();
        const T* p_b = B.data();
              T* p_c = C.data();

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);
        stride_type rs_c = C.stride(0);
        stride_type cs_c = C.stride(1);

        if (m == MR && n == NR)
        {
            if (flip_ukr)
            {
                auxinfo_t aux{p_b, p_a, p_c};
                cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                     &beta, p_c, cs_c, rs_c, &aux);
            }
            else
            {
                auxinfo_t aux{p_a, p_b, p_c};
                cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                     &beta, p_c, rs_c, cs_c, &aux);
            }
        }
        else
        {
            T p_ab[512] __attribute__((aligned(64)));
            static const T zero = T(0);

            if (flip_ukr)
            {
                auxinfo_t aux{p_b, p_a, p_c};
                cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                     &zero, &p_ab[0], cs_ab, rs_ab, &aux);
            }
            else
            {
                auxinfo_t aux{p_a, p_b, p_c};
                cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                     &zero, &p_ab[0], rs_ab, cs_ab, &aux);
            }

            accum_utile(m, n, p_ab, rs_ab, cs_ab,
                        beta, p_c, rs_c, cs_c);
        }
    }

    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha,        normal_matrix<T>& A,
                                    normal_matrix<T>& B,
                    T  beta, block_scatter_matrix<T>& C) const
    {
        (void)comm;

        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const bool flip_ukr = cfg.gemm_flip_ukr.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        const T* p_a = A.data();
        const T* p_b = B.data();
              T* p_c;

        TBLIS_ASSERT(C.block_size(0) == MR);
        TBLIS_ASSERT(C.block_size(1) == NR);

        len_type m, n;
        len_type k = A.length(1);
        stride_type rs_c, cs_c;
        const stride_type *rscat_c, *cscat_c;

        C.block(p_c, rscat_c, rs_c, m, cscat_c, cs_c, n);
        auto c_prefetch = p_c + (rs_c ? 0 : *rscat_c) + (cs_c ? 0 : *cscat_c);

        if (m == MR && n == NR && rs_c && cs_c)
        {
            if (flip_ukr)
            {
                auxinfo_t aux{p_b, p_a, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                     &beta, p_c, cs_c, rs_c, &aux);
            }
            else
            {
                auxinfo_t aux{p_a, p_b, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                     &beta, p_c, rs_c, cs_c, &aux);
            }
        }
        else
        {
            T p_ab[512] __attribute__((aligned(64)));
            static const T zero = T(0);

            if (flip_ukr)
            {
                auxinfo_t aux{p_b, p_a, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                     &zero, &p_ab[0], cs_ab, rs_ab, &aux);
            }
            else
            {
                auxinfo_t aux{p_a, p_b, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                     &zero, &p_ab[0], rs_ab, cs_ab, &aux);
            }

            if (rs_c && cs_c)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cs_c);
            }
            else if (rs_c)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cscat_c);
            }
            else if (cs_c)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cs_c);
            }
            else
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cscat_c);
            }
        }
    }

    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha,              normal_matrix<T>& A,
                                          normal_matrix<T>& B,
                    T  beta, patch_block_scatter_matrix<T>& C) const
    {
        (void)comm;

        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const bool flip_ukr = cfg.gemm_flip_ukr.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        const T* p_a = A.data();
        const T* p_b = B.data();
              T* p_c;

        TBLIS_ASSERT(C.block_size(0) == MR);
        TBLIS_ASSERT(C.block_size(1) == NR);

        len_type m, n;
        len_type k = A.length(1);
        stride_type rs_c, cs_c;
        const stride_type *rscat_c, *cscat_c;

        C.block(p_c, rscat_c, rs_c, m, cscat_c, cs_c, n);
        auto c_prefetch = p_c + (rs_c ? 0 : *rscat_c) + (cs_c ? 0 : *cscat_c);

        if (m == MR && n == NR && rs_c && cs_c)
        {
            if (flip_ukr)
            {
                auxinfo_t aux{p_b, p_a, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                     &beta, p_c, cs_c, rs_c, &aux);
            }
            else
            {
                auxinfo_t aux{p_a, p_b, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                     &beta, p_c, rs_c, cs_c, &aux);
            }
        }
        else
        {
            T p_ab[512] __attribute__((aligned(64)));
            static const T zero = T(0);

            if (flip_ukr)
            {
                auxinfo_t aux{p_b, p_a, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                     &zero, &p_ab[0], cs_ab, rs_ab, &aux);
            }
            else
            {
                auxinfo_t aux{p_a, p_b, c_prefetch};
                cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                     &zero, &p_ab[0], rs_ab, cs_ab, &aux);
            }

            if (rs_c && cs_c)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cs_c);
            }
            else if (rs_c)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cscat_c);
            }
            else if (cs_c)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cs_c);
            }
            else
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cscat_c);
            }
        }
    }
};

}

#endif
