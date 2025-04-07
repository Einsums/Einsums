#ifndef _TBLIS_KERNELS_1M_TRANS_HPP_
#define _TBLIS_KERNELS_1M_TRANS_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/macros.h"

namespace tblis
{

template <typename T>
using trans_ukr_t =
    void (*)(len_type m, len_type n,
             T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
             T  beta, bool conj_B,       T* B, stride_type rs_B, stride_type cs_B);

template <typename Config, typename T>
void trans_ukr_def(len_type m, len_type n,
                   T alpha, bool conj_A, const T* TBLIS_RESTRICT A, stride_type rs_A, stride_type cs_A,
                   T  beta, bool conj_B,       T* TBLIS_RESTRICT B, stride_type rs_B, stride_type cs_B)
{
    constexpr len_type MR = Config::template trans_mr<T>::def;
    constexpr len_type NR = Config::template trans_nr<T>::def;

    if (m == MR && n == NR)
    {
        if (beta == T(0))
        {
            if (is_complex<T>::value && conj_A)
            {
                if (rs_B == 1 && rs_A == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j + i*rs_B] = alpha*conj(A[i + j*cs_A]);
                }
                else if (cs_B == 1 && cs_A == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        for (len_type i = 0;i < MR;i++)
                            B[j*cs_B + i] = alpha*conj(A[i*rs_A + j]);
                }
                else
                {
                     for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j*cs_B + i*rs_B] = alpha*conj(A[i*rs_A + j*cs_A]);
                }
            }
            else
            {
                if (rs_B == 1 && rs_A == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j + i*rs_B] = alpha*A[i + j*cs_A];
                }
                else if (cs_B == 1 && cs_A == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        for (len_type i = 0;i < MR;i++)
                            B[j*cs_B + i] = alpha*A[i*rs_A + j];
                }
                else
                {
                     for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j*cs_B + i*rs_B] = alpha*A[i*rs_A + j*cs_A];
                }
            }
        }
        else
        {
            if (is_complex<T>::value && conj_A && conj_B)
            {
                if (rs_B == 1 && rs_A == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j + i*rs_B] = alpha*conj(A[i + j*cs_A]) +
                                             beta*conj(B[j + i*rs_B]);
                }
                else if (cs_B == 1 && cs_A == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        for (len_type i = 0;i < MR;i++)
                            B[j*cs_B + i] = alpha*conj(A[i*rs_A + j]) +
                                             beta*conj(B[j*cs_B + i]);
                }
                else
                {
                     for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j*cs_B + i*rs_B] = alpha*conj(A[i*rs_A + j*cs_A]) +
                                                  beta*conj(B[j*cs_B + i*rs_B]);
                }
            }
            else if (is_complex<T>::value && conj_B)
            {
                if (rs_B == 1 && rs_A == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j + i*rs_B] = alpha*A[i + j*cs_A] +
                                             beta*conj(B[j + i*rs_B]);
                }
                else if (cs_B == 1 && cs_A == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        for (len_type i = 0;i < MR;i++)
                            B[j*cs_B + i] = alpha*A[i*rs_A + j] +
                                             beta*conj(B[j*cs_B + i]);
                }
                else
                {
                     for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j*cs_B + i*rs_B] = alpha*A[i*rs_A + j*cs_A] +
                                                  beta*conj(B[j*cs_B + i*rs_B]);
                }
            }
            else if (is_complex<T>::value && conj_A)
            {
                if (rs_B == 1 && rs_A == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j + i*rs_B] = alpha*conj(A[i + j*cs_A]) +
                                             beta*B[j + i*rs_B];
                }
                else if (cs_B == 1 && cs_A == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        for (len_type i = 0;i < MR;i++)
                            B[j*cs_B + i] = alpha*conj(A[i*rs_A + j]) +
                                             beta*B[j*cs_B + i];
                }
                else
                {
                     for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j*cs_B + i*rs_B] = alpha*conj(A[i*rs_A + j*cs_A]) +
                                                  beta*B[j*cs_B + i*rs_B];
                }
            }
            else
            {
                if (rs_B == 1 && rs_A == 1)
                {
                    for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j + i*rs_B] = alpha*A[i + j*cs_A] +
                                             beta*B[j + i*rs_B];
                }
                else if (cs_B == 1 && cs_A == 1)
                {
                    for (len_type j = 0;j < NR;j++)
                        for (len_type i = 0;i < MR;i++)
                            B[j*cs_B + i] = alpha*A[i*rs_A + j] +
                                             beta*B[j*cs_B + i];
                }
                else
                {
                     for (len_type i = 0;i < MR;i++)
                        for (len_type j = 0;j < NR;j++)
                            B[j*cs_B + i*rs_B] = alpha*A[i*rs_A + j*cs_A] +
                                                  beta*B[j*cs_B + i*rs_B];
                }
            }
        }
    }
    else
    {
        if (beta == T(0))
        {
             for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    B[j*cs_B + i*rs_B] = alpha*conj(conj_A, A[i*rs_A + j*cs_A]);
        }
        else
        {
             for (len_type i = 0;i < m;i++)
                for (len_type j = 0;j < n;j++)
                    B[j*cs_B + i*rs_B] = alpha*conj(conj_A, A[i*rs_A + j*cs_A]) +
                                          beta*conj(conj_B, B[j*cs_B + i*rs_B]);
        }
    }
}

}

#endif
