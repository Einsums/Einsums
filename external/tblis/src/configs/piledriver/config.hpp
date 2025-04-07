#ifndef _TBLIS_CONFIGS_PILEDRIVER_CONFIG_HPP_
#define _TBLIS_CONFIGS_PILEDRIVER_CONFIG_HPP_

#include "configs/config_builder.hpp"

extern "C"
{

EXTERN_GEMM_UKR(          float, bli_sgemm_asm_16x3);
EXTERN_GEMM_UKR(         double, bli_dgemm_asm_8x3);
EXTERN_GEMM_UKR(tblis::scomplex, bli_cgemm_asm_4x2);
EXTERN_GEMM_UKR(tblis::dcomplex, bli_zgemm_asm_2x2);

}

namespace tblis
{

extern int piledriver_check();

TBLIS_BEGIN_CONFIG(piledriver)

TBLIS_CONFIG_GEMM_MR(  16,    8,    4,    2)
TBLIS_CONFIG_GEMM_NR(   3,    3,    2,    2)
TBLIS_CONFIG_GEMM_KR(   8,    4,    4,    4)
TBLIS_CONFIG_GEMM_MC(2016, 1008,  512,  400)
TBLIS_CONFIG_GEMM_NC(8400, 8400, 8400, 8400)
TBLIS_CONFIG_GEMM_KC( 128,  128,  256,  160)

TBLIS_CONFIG_GEMM_UKR(bli_sgemm_asm_16x3,
                      bli_dgemm_asm_8x3,
                      bli_cgemm_asm_4x2,
                      bli_zgemm_asm_2x2)

TBLIS_CONFIG_CHECK(piledriver_check)

TBLIS_END_CONFIG

}

#endif
