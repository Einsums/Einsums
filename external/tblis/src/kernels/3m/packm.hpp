#ifndef _TBLIS_KERNELS_3M_PACKM_HPP_
#define _TBLIS_KERNELS_3M_PACKM_HPP_

#include "util/basic_types.h"
#include <type_traits>

namespace tblis
{

#define EXTERN_PACK_NN_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               tblis::stride_type cs_a, \
                 T* p_ap);

template <typename T>
using pack_nn_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, stride_type cs_a,
         T* p_ap);

#define EXTERN_PACK_NND_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               tblis::stride_type cs_a, \
                 const T* p_d, tblis::stride_type inc_d, \
                 T* p_ap);

template <typename T>
using pack_nnd_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, stride_type cs_a,
         const T* p_d, stride_type inc_d,
         T* p_ap);

#define EXTERN_PACK_SN_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               tblis::stride_type cs_a, \
                 T* p_ap);

template <typename T>
using pack_sn_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, stride_type cs_a,
         T* p_ap);

#define EXTERN_PACK_NS_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               const tblis::stride_type* cscat_a, \
                 T* p_ap);

template <typename T>
using pack_ns_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, const stride_type* cscat_a,
         T* p_ap);

#define EXTERN_PACK_SS_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               const tblis::stride_type* cscat_a, \
                 T* p_ap);

template <typename T>
using pack_ss_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         T* p_ap);

#define EXTERN_PACK_NB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, tblis::stride_type rs_a, \
                               const tblis::stride_type* cscat_a, \
                               const tblis::stride_type* cbs_a, \
                 T* p_ap);

template <typename T>
using pack_nb_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, stride_type rs_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         T* p_ap);

#define EXTERN_PACK_SB_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               const tblis::stride_type* cscat_a, \
                               const tblis::stride_type* cbs_a, \
                 T* p_ap);

template <typename T>
using pack_sb_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const stride_type* cscat_a,
         const stride_type* cbs_a,
         T* p_ap);

#define EXTERN_PACK_SS_SCAL_UKR(T, name) \
extern void name(tblis::len_type m, tblis::len_type k, \
                 const T* p_a, const tblis::stride_type* rscat_a, \
                               const T* rscale_a, \
                               const tblis::stride_type* cscat_a, \
                               const T* cscale_a, \
                 T* p_ap);

template <typename T>
using pack_ss_scal_ukr_t =
void (*)(len_type m, len_type k,
         const T* p_a, const stride_type* rscat_a, const T* rscale_a,
         const stride_type* cscat_a, const T* cscale_a, T* p_ap);

template <typename Config, typename T, int Mat>
void pack_nn_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a, stride_type rs_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        for (len_type p = 0;p < k;p++)
        {
            #pragma omp simd
            for (len_type mr = 0;mr < MR;mr++)
                p_ap[mr] = p_a[mr];

            p_a += cs_a;
            p_ap += ME;
        }
    }
    else if (m == MR && cs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];

            p_a += KR;
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
            for (len_type mr = 0;mr < MR;mr++)
                p_ap[mr + ME*kr] = p_a[rs_a*mr + kr];
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
                p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p];

            for (len_type mr = m;mr < MR;mr++)
                p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_nnd_ukr_def(len_type m, len_type k,
                      const T* TBLIS_RESTRICT p_a, stride_type rs_a, stride_type cs_a,
                      const T* TBLIS_RESTRICT p_d, stride_type inc_d,
                      T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        for (len_type p = 0;p < k;p++)
        {
            #pragma omp simd
            for (len_type mr = 0;mr < MR;mr++)
                p_ap[mr] = p_a[mr] * (*p_d);

            p_a += cs_a;
            p_d += inc_d;
            p_ap += ME;
        }
    }
    else if (m == MR && cs_a == 1)
    {
        len_type p = 0;
        for (;p < k-KR;p += KR)
        {
            for (len_type kr = 0;kr < KR;kr++)
                for (len_type mr = 0;mr < MR;mr++)
                    p_ap[mr + ME*kr] = p_a[rs_a*mr + kr] * p_d[kr*inc_d];

            p_a += KR;
            p_d += inc_d*KR;
            p_ap += ME*KR;
        }

        for (len_type kr = 0;kr < k-p;kr++)
            for (len_type mr = 0;mr < MR;mr++)
                p_ap[mr + ME*kr] = p_a[rs_a*mr + kr] * p_d[kr*inc_d];
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
                p_ap[mr + ME*p] = p_a[rs_a*mr + cs_a*p] * p_d[inc_d*p];

            for (len_type mr = m;mr < MR;mr++)
                p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_sn_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     const stride_type* TBLIS_RESTRICT rscat_a, stride_type cs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cs_a*p];

        for (len_type mr = m;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
    }
}

template <typename Config, typename T, int Mat>
void pack_ns_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
            p_ap[mr + ME*p] = p_a[rs_a*mr + cscat_a[p]];

        for (len_type mr = m;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
    }
}

template <typename Config, typename T, int Mat>
void pack_ss_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     const stride_type* TBLIS_RESTRICT rscat_a,
                     const stride_type* TBLIS_RESTRICT cscat_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];

        for (len_type mr = m;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
    }
}

template <typename Config, typename T, int Mat>
void pack_nb_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     stride_type rs_a, const stride_type* TBLIS_RESTRICT cscat_a,
                     const stride_type* TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);
    constexpr len_type KR = Config::template gemm_kr<T>::def;

    if (m == MR && rs_a == 1)
    {
        for (len_type p = 0;p < k;p += KR)
        {
            len_type k_loc = std::min(KR, k-p);
            stride_type cs_a = *cbs_a;
            stride_type off_a = *cscat_a;

            if (cs_a)
            {
                for (len_type kr = 0;kr < k_loc;kr++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*kr] = p_a[mr + cs_a*kr + off_a];
            }
            else
            {
                for (len_type kr = 0;kr < k_loc;kr++)
                    #pragma omp simd
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*kr] = p_a[mr + cscat_a[kr]];
            }

            p_ap += ME*KR;
            cscat_a += KR;
            cbs_a += KR;
        }
    }
    else if (m == MR)
    {
        for (len_type p = 0;p < k;p += KR)
        {
            len_type k_loc = std::min(KR, k-p);
            stride_type cs_a = *cbs_a;
            stride_type off_a = *cscat_a;

            if (cs_a == 1)
            {
                for (len_type kr = 0;kr < k_loc;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*kr] = p_a[rs_a*mr + kr + off_a];
            }
            else if (cs_a)
            {
                for (len_type kr = 0;kr < k_loc;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*kr] = p_a[rs_a*mr + cs_a*kr + off_a];
            }
            else
            {
                for (len_type kr = 0;kr < k_loc;kr++)
                    for (len_type mr = 0;mr < MR;mr++)
                        p_ap[mr + ME*kr] = p_a[rs_a*mr + cscat_a[kr]];
            }

            p_ap += ME*KR;
            cscat_a += KR;
            cbs_a += KR;
        }
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
                p_ap[mr + ME*p] = p_a[rs_a*mr + cscat_a[p]];

            for (len_type mr = m;mr < MR;mr++)
                p_ap[mr + ME*p] = T();
        }
    }
}

template <typename Config, typename T, int Mat>
void pack_sb_ukr_def(len_type m, len_type k,
                     const T* TBLIS_RESTRICT p_a,
                     const stride_type* TBLIS_RESTRICT rscat_a,
                     const stride_type* TBLIS_RESTRICT cscat_a,
                     const stride_type* TBLIS_RESTRICT cbs_a,
                     T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    (void)cbs_a;

    for (len_type p = 0;p < k;p++)
    {
        for (len_type mr = 0;mr < m;mr++)
            p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]];

        for (len_type mr = m;mr < MR;mr++)
            p_ap[mr + ME*p] = T();
    }
}

template <typename Config, typename T, int Mat>
void pack_ss_scal_ukr_def(len_type m, len_type k,
                          const T* TBLIS_RESTRICT p_a,
                          const stride_type* TBLIS_RESTRICT rscat_a,
                          const T* TBLIS_RESTRICT rscale_a,
                          const stride_type* TBLIS_RESTRICT cscat_a,
                          const T* TBLIS_RESTRICT cscale_a,
                          T* TBLIS_RESTRICT p_ap)
{
    using namespace matrix_constants;
    constexpr len_type MR = (Mat == MAT_A ? Config::template gemm_mr<T>::def
                                          : Config::template gemm_nr<T>::def);
    constexpr len_type ME = (Mat == MAT_A ? Config::template gemm_mr<T>::extent
                                          : Config::template gemm_nr<T>::extent);

    if (m == MR)
    {
        for (len_type p = 0;p < k;p++)
            for (len_type mr = 0;mr < MR;mr++)
                p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]] * rscale_a[mr] * cscale_a[p];
    }
    else
    {
        for (len_type p = 0;p < k;p++)
        {
            for (len_type mr = 0;mr < m;mr++)
                p_ap[mr + ME*p] = p_a[rscat_a[mr] + cscat_a[p]] * rscale_a[mr] * cscale_a[p];

            for (len_type mr = m;mr < MR;mr++)
                p_ap[mr + ME*p] = T();
        }
    }
}

}

#endif
