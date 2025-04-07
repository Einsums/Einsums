/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include <assert.h>

#include "../knl/bli_avx512_macros.h"

#define A_L1_PREFETCH_DIST 4 //should be multiple of 4

#define LOOP_ALIGN ALIGN16

#define UPDATE_C(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VFMADD231PD(ZMM(R1), ZMM(1), MEM(RCX,0*64)) \
    VFMADD231PD(ZMM(R2), ZMM(1), MEM(RCX,1*64)) \
    VFMADD231PD(ZMM(R3), ZMM(1), MEM(RCX,2*64)) \
    VFMADD231PD(ZMM(R4), ZMM(1), MEM(RCX,3*64)) \
    VMOVUPD(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPD(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,2*64), ZMM(R3)) \
    VMOVUPD(MEM(RCX,3*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_BZ(R1,R2,R3,R4) \
\
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VMOVUPD(MEM(RCX,0*64), ZMM(R1)) \
    VMOVUPD(MEM(RCX,1*64), ZMM(R2)) \
    VMOVUPD(MEM(RCX,2*64), ZMM(R3)) \
    VMOVUPD(MEM(RCX,3*64), ZMM(R4)) \
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(2),8)) \
    VFMADD231PD(ZMM(R1), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(2), ZMM(R1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(3),8)) \
    VFMADD231PD(ZMM(R2), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),8) MASK_K(2), ZMM(R2)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(4),8)) \
    VFMADD231PD(ZMM(R3), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),8) MASK_K(2), ZMM(R3)) \
\
    KXNORW(K(1), K(0), K(0)) \
    KXNORW(K(2), K(0), K(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VGATHERQPD(ZMM(6) MASK_K(1), MEM(RCX,ZMM(5),8)) \
    VFMADD231PD(ZMM(R4), ZMM(6), ZMM(1)) \
    VSCATTERQPD(MEM(RCX,ZMM(5),8) MASK_K(2), ZMM(R4)) \
\
    LEA(RCX, MEM(RCX,RAX,1))

#define UPDATE_C_BZ_ROW_SCATTERED(R1,R2,R3,R4) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R1), ZMM(R1), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(2),8) MASK_K(1), ZMM(R1)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R2), ZMM(R2), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(3),8) MASK_K(1), ZMM(R2)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R3), ZMM(R3), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(4),8) MASK_K(1), ZMM(R3)) \
\
    KXNORW(K(1), K(0), K(0)) \
    VMULPD(ZMM(R4), ZMM(R4), ZMM(0)) \
    VSCATTERQPD(MEM(RCX,ZMM(5),8) MASK_K(1), ZMM(R4)) \
\
    LEA(RCX, MEM(RCX,RAX,1))

#define PREFETCH_C_L1 \
\
    PREFETCHW0(MEM(RCX,      0*64)) \
    PREFETCHW0(MEM(RCX,      1*64)) \
    PREFETCHW0(MEM(RCX,      2*64)) \
    PREFETCHW0(MEM(RCX,      3*64)) \
    PREFETCHW0(MEM(RCX,R12,1,0*64)) \
    PREFETCHW0(MEM(RCX,R12,1,1*64)) \
    PREFETCHW0(MEM(RCX,R12,1,2*64)) \
    PREFETCHW0(MEM(RCX,R12,1,3*64)) \
    PREFETCHW0(MEM(RCX,R12,2,0*64)) \
    PREFETCHW0(MEM(RCX,R12,2,1*64)) \
    PREFETCHW0(MEM(RCX,R12,2,2*64)) \
    PREFETCHW0(MEM(RCX,R12,2,3*64)) \
    PREFETCHW0(MEM(RCX,R13,1,0*64)) \
    PREFETCHW0(MEM(RCX,R13,1,1*64)) \
    PREFETCHW0(MEM(RCX,R13,1,2*64)) \
    PREFETCHW0(MEM(RCX,R13,1,3*64)) \
    PREFETCHW0(MEM(RCX,R12,4,0*64)) \
    PREFETCHW0(MEM(RCX,R12,4,1*64)) \
    PREFETCHW0(MEM(RCX,R12,4,2*64)) \
    PREFETCHW0(MEM(RCX,R12,4,3*64)) \
    PREFETCHW0(MEM(RCX,R14,1,0*64)) \
    PREFETCHW0(MEM(RCX,R14,1,1*64)) \
    PREFETCHW0(MEM(RCX,R14,1,2*64)) \
    PREFETCHW0(MEM(RCX,R14,1,3*64)) \

//
// n: index in unrolled loop
//
// a: ZMM register to load into
// b: ZMM register to read from
//
// ...: addressing for A, except for offset
//
#define SUBITER(n) \
\
    VBROADCASTSD(ZMM(4), MEM(RAX,(6*n+0)*8)) \
    VBROADCASTSD(ZMM(5), MEM(RAX,(6*n+1)*8)) \
    VFMADD231PD(ZMM( 8), ZMM(0), ZMM(4))  VFMADD231PD(ZMM(12), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM( 9), ZMM(1), ZMM(4))  VFMADD231PD(ZMM(13), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(10), ZMM(2), ZMM(4))  VFMADD231PD(ZMM(14), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(11), ZMM(3), ZMM(4))  VFMADD231PD(ZMM(15), ZMM(3), ZMM(5)) \
    \
    VBROADCASTSD(ZMM(4), MEM(RAX,(6*n+2)*8)) \
    VBROADCASTSD(ZMM(5), MEM(RAX,(6*n+3)*8)) \
    VFMADD231PD(ZMM(16), ZMM(0), ZMM(4))  VFMADD231PD(ZMM(20), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(17), ZMM(1), ZMM(4))  VFMADD231PD(ZMM(21), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(18), ZMM(2), ZMM(4))  VFMADD231PD(ZMM(22), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(19), ZMM(3), ZMM(4))  VFMADD231PD(ZMM(23), ZMM(3), ZMM(5)) \
    \
    VBROADCASTSD(ZMM(4), MEM(RAX,(6*n+4)*8)) \
    VBROADCASTSD(ZMM(5), MEM(RAX,(6*n+5)*8)) \
    VFMADD231PD(ZMM(24), ZMM(0), ZMM(4))  VFMADD231PD(ZMM(28), ZMM(0), ZMM(5)) \
    VFMADD231PD(ZMM(25), ZMM(1), ZMM(4))  VFMADD231PD(ZMM(29), ZMM(1), ZMM(5)) \
    VFMADD231PD(ZMM(26), ZMM(2), ZMM(4))  VFMADD231PD(ZMM(30), ZMM(2), ZMM(5)) \
    VFMADD231PD(ZMM(27), ZMM(3), ZMM(4))  VFMADD231PD(ZMM(31), ZMM(3), ZMM(5)) \
    \
    VMOVAPD(ZMM(0), MEM(RBX,(32*n+ 0)*8)) \
    VMOVAPD(ZMM(1), MEM(RBX,(32*n+ 8)*8)) \
    VMOVAPD(ZMM(2), MEM(RBX,(32*n+16)*8)) \
    VMOVAPD(ZMM(3), MEM(RBX,(32*n+24)*8))

//This is an array used for the scatter/gather instructions.
static int64_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

void bli_dgemm_opt_6x32_l1(
                             dim_t            k_,
                             double* restrict alpha,
                             double* restrict a,
                             double* restrict b,
                             double* restrict beta,
                             double* restrict c, inc_t rs_c_, inc_t cs_c_,
                             auxinfo_t*       data,
                             cntx_t* restrict cntx
                           )
{
    (void)data;
    (void)cntx;

    const int64_t* offsetPtr = &offsets[0];
    const int64_t k = k_;
    const int64_t rs_c = rs_c_;
    const int64_t cs_c = cs_c_;

    __asm__ volatile
    (

    VXORPD(YMM(8), YMM(8), YMM(8)) //clear out registers
    VMOVAPD(YMM( 7), YMM(8))
    VMOVAPD(YMM( 9), YMM(8))
    VMOVAPD(YMM(10), YMM(8))   MOV(RSI, VAR(k)) //loop index
    VMOVAPD(YMM(11), YMM(8))   MOV(RAX, VAR(a)) //load address of a
    VMOVAPD(YMM(12), YMM(8))   MOV(RBX, VAR(b)) //load address of b
    VMOVAPD(YMM(13), YMM(8))   MOV(RCX, VAR(c)) //load address of c
    VMOVAPD(YMM(14), YMM(8))
    VMOVAPD(YMM(15), YMM(8))   VMOVAPD(ZMM(0), MEM(RBX, 0*8)) //pre-load b
    VMOVAPD(YMM(16), YMM(8))   VMOVAPD(ZMM(1), MEM(RBX, 8*8)) //pre-load b
    VMOVAPD(YMM(17), YMM(8))   VMOVAPD(ZMM(2), MEM(RBX,16*8)) //pre-load b
    VMOVAPD(YMM(18), YMM(8))   VMOVAPD(ZMM(3), MEM(RBX,24*8)) //pre-load b
    VMOVAPD(YMM(19), YMM(8))
    VMOVAPD(YMM(20), YMM(8))
    VMOVAPD(YMM(21), YMM(8))   MOV(R12, VAR(rs_c))      //rs_c
    VMOVAPD(YMM(22), YMM(8))   LEA(R13, MEM(R12,R12,2)) //*3
    VMOVAPD(YMM(23), YMM(8))   LEA(R14, MEM(R12,R12,4)) //*5
    VMOVAPD(YMM(24), YMM(8))
    VMOVAPD(YMM(25), YMM(8))   MOV(R8, IMM( 6*8)) //mr*sizeof(double)
    VMOVAPD(YMM(26), YMM(8))   MOV(R9, IMM(32*8)) //nr*sizeof(double)
    VMOVAPD(YMM(27), YMM(8))
    VMOVAPD(YMM(28), YMM(8))   LEA(RBX, MEM(RBX,R9,1)) //adjust b for pre-load
    VMOVAPD(YMM(29), YMM(8))
    VMOVAPD(YMM(30), YMM(8))
    VMOVAPD(YMM(31), YMM(8))

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    PREFETCH_C_L1

    MOV(RDI, RSI)
    AND(RSI, IMM(3))
    SAR(RDI, IMM(2))
    JZ(TAIL_LOOP)

    LOOP_ALIGN
    LABEL(MAIN_LOOP)

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*6*8))
        SUBITER(0)
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*6*8+64))
        SUBITER(1)
        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*6*8+128))
        SUBITER(2)
        SUBITER(3)

        LEA(RAX, MEM(RAX,R8,4))
        LEA(RBX, MEM(RBX,R9,4))

        DEC(RDI)

    JNZ(MAIN_LOOP)

    TEST(RSI, RSI)
    JZ(POSTACCUM)

    LOOP_ALIGN
    LABEL(TAIL_LOOP)

        PREFETCH(0, MEM(RAX,A_L1_PREFETCH_DIST*6*8))
        SUBITER(0)

        ADD(RAX, R8)
        ADD(RBX, R9)

        DEC(RSI)

    JNZ(TAIL_LOOP)

    LABEL(POSTACCUM)

    MOV(RAX, VAR(alpha))
    MOV(RBX, VAR(beta))
    VBROADCASTSD(ZMM(0), MEM(RAX))
    VBROADCASTSD(ZMM(1), MEM(RBX))

    MOV(RAX, VAR(rs_c))
    LEA(RAX, MEM(,RAX,8))
    MOV(RBX, VAR(cs_c))

    // Check if C is row stride. If not, jump to the slow scattered update
    CMP(RBX, IMM(1))
    JNE(SCATTEREDUPDATE)

        VCOMISD(XMM(1), XMM(7))
        JE(COLSTORBZ)

            UPDATE_C( 8, 9,10,11)
            UPDATE_C(12,13,14,15)
            UPDATE_C(16,17,18,19)
            UPDATE_C(20,21,22,23)
            UPDATE_C(24,25,26,27)
            UPDATE_C(28,29,30,31)

        JMP(END)
        LABEL(COLSTORBZ)

            UPDATE_C_BZ( 8, 9,10,11)
            UPDATE_C_BZ(12,13,14,15)
            UPDATE_C_BZ(16,17,18,19)
            UPDATE_C_BZ(20,21,22,23)
            UPDATE_C_BZ(24,25,26,27)
            UPDATE_C_BZ(28,29,30,31)

    JMP(END)
    LABEL(SCATTEREDUPDATE)

        MOV(RDI, VAR(offsetPtr))
        VMOVDQA64(ZMM(2), MEM(RDI,0*64))
        VMOVDQA64(ZMM(3), MEM(RDI,1*64))
        VMOVDQA64(ZMM(4), MEM(RDI,2*64))
        VMOVDQA64(ZMM(5), MEM(RDI,3*64))
        VPBROADCASTQ(ZMM(6), RBX)
        VPMULLQ(ZMM(2), ZMM(6), ZMM(2))
        VPMULLQ(ZMM(3), ZMM(6), ZMM(3))
        VPMULLQ(ZMM(4), ZMM(6), ZMM(4))
        VPMULLQ(ZMM(5), ZMM(6), ZMM(5))

        VCOMISD(XMM(1), XMM(7))
        JE(SCATTERBZ)

            UPDATE_C_ROW_SCATTERED( 8, 9,10,11)
            UPDATE_C_ROW_SCATTERED(12,13,14,15)
            UPDATE_C_ROW_SCATTERED(16,17,18,19)
            UPDATE_C_ROW_SCATTERED(20,21,22,23)
            UPDATE_C_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_ROW_SCATTERED(28,29,30,31)

        JMP(END)
        LABEL(SCATTERBZ)

            UPDATE_C_BZ_ROW_SCATTERED( 8, 9,10,11)
            UPDATE_C_BZ_ROW_SCATTERED(12,13,14,15)
            UPDATE_C_BZ_ROW_SCATTERED(16,17,18,19)
            UPDATE_C_BZ_ROW_SCATTERED(20,21,22,23)
            UPDATE_C_BZ_ROW_SCATTERED(24,25,26,27)
            UPDATE_C_BZ_ROW_SCATTERED(28,29,30,31)

    LABEL(END)

    VZEROUPPER()

    : // output operands
    : // input operands
      [k]         "m" (k),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c),
      [offsetPtr] "m" (offsetPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    );
}
