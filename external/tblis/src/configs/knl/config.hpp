#ifndef _TBLIS_CONFIGS_KNL_CONFIG_HPP_
#define _TBLIS_CONFIGS_KNL_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR( float, bli_sgemm_opt_30x16_knc);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_30x8_knc);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_30x8);
EXTERN_GEMM_UKR( float, bli_sgemm_opt_24x16);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_24x8);
EXTERN_GEMM_UKR(double, bli_dgemm_opt_8x24);

}

namespace tblis
{

EXTERN_PACK_NN_UKR(double, knl_dpackm_30xk);
EXTERN_PACK_NN_UKR( float, knl_spackm_24xk);
EXTERN_PACK_NN_UKR(double, knl_dpackm_24xk);
EXTERN_PACK_NN_UKR( float, knl_spackm_16xk);
EXTERN_PACK_NN_UKR(double, knl_dpackm_8xk);

extern int knl_check();

TBLIS_BEGIN_CONFIG(knl_d30x8_knc)

    TBLIS_CONFIG_GEMM_MR_EXTENT(   30,    30, _, _,
                                   32,    32, _, _)
    TBLIS_CONFIG_GEMM_NR       (   16,     8, _, _)
    TBLIS_CONFIG_GEMM_KR       (   16,     8, _, _)
    TBLIS_CONFIG_GEMM_MC_MAX   (  240,   120, _, _,
                                  300,   150, _, _)
    TBLIS_CONFIG_GEMM_NC       (14400, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX   (  240,   240, _, _,
                                  300,   300, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_30x16_knc,
                          bli_dgemm_opt_30x8_knc,
                          _,
                          _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, _, _)

    TBLIS_CONFIG_CHECK(knl_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(knl_d30x8)

    TBLIS_CONFIG_GEMM_MR_EXTENT(_,    30, _, _,
                                _,    32, _, _)
    TBLIS_CONFIG_GEMM_NR       (_,     8, _, _)
    TBLIS_CONFIG_GEMM_KR       (_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC_MAX   (_,   120, _, _,
                                _,   150, _, _)
    TBLIS_CONFIG_GEMM_NC       (_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX   (_,   240, _, _,
                                _,   300, _, _)

    TBLIS_CONFIG_GEMM_UKR(_, bli_dgemm_opt_30x8, _, _)

    TBLIS_CONFIG_PACK_NN_MR_UKR(_, knl_dpackm_30xk, _, _)
    TBLIS_CONFIG_PACK_NN_NR_UKR(_, knl_dpackm_8xk , _, _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(_, true, _, _)

    TBLIS_CONFIG_CHECK(knl_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(knl_d24x8)

    TBLIS_CONFIG_GEMM_MR    (   24,    24, _, _)
    TBLIS_CONFIG_GEMM_NR    (   16,     8, _, _)
    TBLIS_CONFIG_GEMM_KR    (   16,     8, _, _)
    TBLIS_CONFIG_GEMM_MC    (  240,   120, _, _)
    TBLIS_CONFIG_GEMM_NC    (14400, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX(  336,   336, _, _,
                               408,   408, _, _)

    TBLIS_CONFIG_GEMM_UKR(bli_sgemm_opt_24x16, bli_dgemm_opt_24x8 , _, _)

    TBLIS_CONFIG_PACK_NN_MR_UKR(knl_spackm_24xk, knl_dpackm_24xk, _, _)
    TBLIS_CONFIG_PACK_NN_NR_UKR(knl_spackm_16xk, knl_dpackm_8xk , _, _)

    TBLIS_CONFIG_GEMM_ROW_MAJOR(true, true, _, _)

    TBLIS_CONFIG_M_THREAD_RATIO(4, 4, _, _)
    TBLIS_CONFIG_NR_MAX_THREAD(1, 1, _, _)

    TBLIS_CONFIG_CHECK(knl_check)

TBLIS_END_CONFIG

TBLIS_BEGIN_CONFIG(knl_d8x24)

    TBLIS_CONFIG_GEMM_MR    (_,     8, _, _)
    TBLIS_CONFIG_GEMM_NR    (_,    24, _, _)
    TBLIS_CONFIG_GEMM_KR    (_,     8, _, _)
    TBLIS_CONFIG_GEMM_MC    (_,   120, _, _)
    //TBLIS_CONFIG_GEMM_MC_MAX(_,   120, _, _,
    //                         _,   144, _, _)
    TBLIS_CONFIG_GEMM_NC    (_, 14400, _, _)
    TBLIS_CONFIG_GEMM_KC_MAX(_,   336, _, _,
                             _,   408, _, _)

    TBLIS_CONFIG_GEMM_UKR(_, bli_dgemm_opt_8x24, _, _)

    TBLIS_CONFIG_PACK_NN_MR_UKR(_, knl_dpackm_8xk, _, _)
    TBLIS_CONFIG_PACK_NN_NR_UKR(_, knl_dpackm_24xk , _, _)

    TBLIS_CONFIG_M_THREAD_RATIO(_, 16, _, _)
    TBLIS_CONFIG_NR_MAX_THREAD(_, 1, _, _)

    TBLIS_CONFIG_CHECK(knl_check)

TBLIS_END_CONFIG

typedef knl_d24x8_config knl_config;

}

#endif
