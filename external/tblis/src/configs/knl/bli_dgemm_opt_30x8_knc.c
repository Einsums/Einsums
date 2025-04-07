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

#include "bli_avx512_macros.h"

#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 2
#define L2_PREFETCH_DIST  16 // Must be greater than 10, because of the way the loop is constructed.

//Alternate code path uused if C is not row-major
// r9 = c
// ymm0 = cs_c * 1...8
// r11 = rs_c
// r12 = &alpha
// r13 = &beta
#define UPDATE_C_ROW_SCATTERED_(NUM,BNZ1,BNZ2) \
\
    BNZ1 KXNORW(K(2), K(0), K(0)) BNZ2 \
    KXNORW(K(3), K(0), K(0)) \
    BNZ1 VGATHERDPD(ZMM(31) MASK_K(2), MEM(R(9),YMM(0),8)) BNZ2 \
    VMULPD(ZMM(NUM), ZMM(NUM), MEM_1TO8(R(12))) /*scale by alpha*/ \
    BNZ1 VFMADD231PD(ZMM(NUM), ZMM(31), MEM_1TO8(R(13))) BNZ2 /*scale by beta, add in result*/ \
    VSCATTERDPD(MEM(R(9),YMM(0),8) MASK_K(3), ZMM(NUM)) \
    ADD(R(9), R(11))

#define UPDATE_C_ROW_SCATTERED(NUM) UPDATE_C_ROW_SCATTERED_(NUM,,)
#define UPDATE_C_BZ_ROW_SCATTERED(NUM) UPDATE_C_ROW_SCATTERED_(NUM,COMMENT_BEGIN,COMMENT_END)

// r12 = &alpha
// zmm31 = beta
// r9 = c
// r11 =   rs_c
// r10 = 3*rs_c
// rdi = 4*rs_c
#define UPDATE_C_4_ROWS_(R1,R2,R3,R4,BNZ1,BNZ2) \
\
    VMULPD(ZMM(R1), ZMM(R1), MEM_1TO8(R(12))) \
    VMULPD(ZMM(R2), ZMM(R2), MEM_1TO8(R(12))) \
    VMULPD(ZMM(R3), ZMM(R3), MEM_1TO8(R(12))) \
    VMULPD(ZMM(R4), ZMM(R4), MEM_1TO8(R(12))) \
    BNZ1 VFMADD231PD(ZMM(R1), ZMM(31), MEM(R(9)        )) BNZ2 \
    BNZ1 VFMADD231PD(ZMM(R2), ZMM(31), MEM(R(9),R(11),1)) BNZ2 \
    BNZ1 VFMADD231PD(ZMM(R3), ZMM(31), MEM(R(9),R(11),2)) BNZ2 \
    BNZ1 VFMADD231PD(ZMM(R4), ZMM(31), MEM(R(9),R(10),1)) BNZ2 \
    VMOVUPD(MEM(R(9)        ), ZMM(R1)) \
    VMOVUPD(MEM(R(9),R(11),1), ZMM(R2)) \
    VMOVUPD(MEM(R(9),R(11),2), ZMM(R3)) \
    VMOVUPD(MEM(R(9),R(10),1), ZMM(R4)) \
    ADD(R(9), RDI)

// r12 = &alpha
// zmm31 = beta
// r9 = c
// r11 = rs_c
#define UPDATE_C_2_ROWS_(R1,R2,BNZ1,BNZ2) \
\
    VMULPD(ZMM(R1), ZMM(R1), MEM_1TO8(R(12))) \
    VMULPD(ZMM(R2), ZMM(R2), MEM_1TO8(R(12))) \
    BNZ1 VFMADD231PD(ZMM(R1), ZMM(31), MEM(R(9)        )) BNZ2 \
    BNZ1 VFMADD231PD(ZMM(R2), ZMM(31), MEM(R(9),R(11),1)) BNZ2 \
    VMOVUPD(MEM(R(9)        ), ZMM(R1)) \
    VMOVUPD(MEM(R(9),R(11),1), ZMM(R2)) \

#define UPDATE_C_4_ROWS(R1,R2,R3,R4) UPDATE_C_4_ROWS_(R1,R2,R3,R4,,)
#define UPDATE_C_2_ROWS(R1,R2) UPDATE_C_2_ROWS_(R1,R2,,)
#define UPDATE_C_BZ_4_ROWS(R1,R2,R3,R4) UPDATE_C_4_ROWS_(R1,R2,R3,R4,COMMENT_BEGIN,COMMENT_END)
#define UPDATE_C_BZ_2_ROWS(R1,R2) UPDATE_C_2_ROWS_(R1,R2,COMMENT_BEGIN,COMMENT_END)

#define A_TIMES_B_ROW(n) VFMADD231PD(ZMM(n), ZMM(31), MEM_1TO8(R(15),(n-1)*8))
#define A_TIMES_B_ROW_PREV(n) VFMADD231PD(ZMM(n), ZMM(31), MEM_1TO8(R(15),((n-1)-32)*8))
#define PREFETCH_A_L1(n) PREFETCH(0, MEM(R(15),A_L1_PREFETCH_DIST*8*32+n*64))
#define PREFETCH_A_L2(n) PREFETCH(1, MEM(R(15),R(14),1,n*64))
#define PREFETCH_B_L1 PREFETCH(0, MEM(RBX,B_L1_PREFETCH_DIST*8*8))
#define PREFETCH_B_L2 PREFETCH(1, MEM(RBX,R(13),1))

//One iteration of the k_r loop.
//Each iteration, we prefetch A into L1 and into L2
// r15 = a
// rbx = b
// rcx = c
// r11 = rs_c
// r13 = L2_PREFETCH_DIST*8*8
// r14 = L2_PREFETCH_DIST*8*32
// r12 = 32*8 = dist. to next sliver of a
// r9  =  8*8 = dist. to next sliver of b
#define MAIN_LOOP_(COUNTER, PC_L1_1, PC_L1_2, PC_L2_1, PC_L2_2) \
\
    /* Can this be pre-loaded for next it. in zmm0? */               \
    VMOVAPD(ZMM(31), MEM(RBX))                                       \
                                                                     \
    A_TIMES_B_ROW     ( 1)                                           \
    A_TIMES_B_ROW     ( 2)    PREFETCH_A_L1(0)                       \
    A_TIMES_B_ROW     ( 3)    PREFETCH_A_L1(1)                       \
    A_TIMES_B_ROW     ( 4)    PREFETCH_A_L1(2)                       \
    A_TIMES_B_ROW     ( 5)    PREFETCH_A_L1(3)                       \
    A_TIMES_B_ROW     ( 6)    PREFETCH_A_L2(0)                       \
    A_TIMES_B_ROW     ( 7)    PC_L1_1 PREFETCH(0, MEM(RCX))  PC_L1_2 \
    A_TIMES_B_ROW     ( 8)    PC_L1_1 ADD(RCX, R(11))        PC_L1_2 \
    A_TIMES_B_ROW     ( 9)                                           \
    A_TIMES_B_ROW     (10)    PC_L2_1 PREFETCH(1, MEM(RCX))  PC_L2_2 \
    A_TIMES_B_ROW     (11)    PREFETCH_A_L2(1)                       \
    A_TIMES_B_ROW     (12)    PC_L1_1 PREFETCH(0, MEM(RCX))  PC_L1_2 \
    A_TIMES_B_ROW     (13)    PC_L1_1 ADD(RCX, R(11))        PC_L1_2 \
    A_TIMES_B_ROW     (14)                                           \
    A_TIMES_B_ROW     (15)                                           \
    A_TIMES_B_ROW     (16)    PREFETCH_A_L2(2)                       \
    A_TIMES_B_ROW     (17)    PC_L1_1 PREFETCH(0, MEM(RCX))  PC_L1_2 \
    A_TIMES_B_ROW     (18)    PC_L1_1 ADD(RCX, R(11))        PC_L1_2 \
    A_TIMES_B_ROW     (19)                                           \
    A_TIMES_B_ROW     (20)                                           \
    A_TIMES_B_ROW     (21)    PREFETCH_A_L2(3)                       \
    A_TIMES_B_ROW     (22)    ADD(R(15), R(12))                      \
    A_TIMES_B_ROW_PREV(23)                                           \
    A_TIMES_B_ROW_PREV(24)    PC_L2_1 ADD(RCX, R(11))        PC_L2_2 \
    A_TIMES_B_ROW_PREV(25)    DEC(COUNTER)                           \
    A_TIMES_B_ROW_PREV(26)    PREFETCH_B_L2                          \
    A_TIMES_B_ROW_PREV(27)    PREFETCH_B_L1                          \
    A_TIMES_B_ROW_PREV(28)    ADD(RBX, R(9))                         \
    A_TIMES_B_ROW_PREV(29)    CMP(COUNTER, IMM(0))                   \
    A_TIMES_B_ROW_PREV(30)

#define MAIN_LOOP(COUNTER) MAIN_LOOP_(COUNTER,COMMENT_BEGIN,COMMENT_END,COMMENT_BEGIN,COMMENT_END)
#define MAIN_LOOP_PC_L1(COUNTER) MAIN_LOOP_(COUNTER,,,COMMENT_BEGIN,COMMENT_END)
#define MAIN_LOOP_PC_L2(COUNTER) MAIN_LOOP_(COUNTER,COMMENT_BEGIN,COMMENT_END,,)

//This is an array used for the scatter/gather instructions.
static int32_t offsets[32] __attribute__((aligned(64))) =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
     16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

//#define MONITORS
//#define LOOPMON
void bli_dgemm_opt_30x8_knc(
                    dim_t            k,
                    double* restrict alpha,
                    double* restrict a,
                    double* restrict b,
                    double* restrict beta,
                    double* restrict c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t*       data,
                    cntx_t* restrict cntx
                  )
{
    const double * a_next = bli_auxinfo_next_a( data );
    const double * b_next = bli_auxinfo_next_b( data );

    const int32_t * offsetPtr = &offsets[0];

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif
    
    __asm__ volatile
    (
#ifdef MONITORS
    RDTSC
    MOV(VAR(topl), EAX)
    MOV(VAR(toph), EDX)
#endif

    VPXORD(ZMM(1), ZMM(1), ZMM(1)) //clear out registers

    VMOVAPS(ZMM( 2), ZMM(1))
    VMOVAPS(ZMM( 3), ZMM(1))    MOV(RSI, VAR(k)) //loop index
    VMOVAPS(ZMM( 4), ZMM(1))    MOV(R(11), VAR(rs_c)) //load row stride
    VMOVAPS(ZMM( 5), ZMM(1))    SAL(R(11), IMM(3)) //scale row stride
    VMOVAPS(ZMM( 6), ZMM(1))    MOV(R(15), VAR(a)) //load address of a
    VMOVAPS(ZMM( 7), ZMM(1))    MOV(RBX, VAR(b)) //load address of b
    VMOVAPS(ZMM( 8), ZMM(1))
    VMOVAPS(ZMM( 9), ZMM(1))    LEA(R(10), MEM(R(11),R(11),2)) //r10 has 3 * r11
    VMOVAPS(ZMM(10), ZMM(1))
    VMOVAPS(ZMM(11), ZMM(1))    MOV(RDI, R(11))
    VMOVAPS(ZMM(12), ZMM(1))    SAL(RDI, IMM(2)) //rdi has 4*r11
    VMOVAPS(ZMM(13), ZMM(1))    MOV(RCX, VAR(c)) //load address of c for prefetching
    VMOVAPS(ZMM(14), ZMM(1))
    VMOVAPS(ZMM(15), ZMM(1))    MOV(R(8), VAR(k))
    VMOVAPS(ZMM(16), ZMM(1))
    VMOVAPS(ZMM(17), ZMM(1))
    VMOVAPS(ZMM(18), ZMM(1))    MOV(R(13), IMM(8*8*L2_PREFETCH_DIST))
    VMOVAPS(ZMM(19), ZMM(1))    MOV(R(14), IMM(8*32*L2_PREFETCH_DIST))
    VMOVAPS(ZMM(20), ZMM(1))
    VMOVAPS(ZMM(21), ZMM(1))
    VMOVAPS(ZMM(22), ZMM(1))
    VMOVAPS(ZMM(23), ZMM(1))
    VMOVAPS(ZMM(24), ZMM(1))    SUB(R(8), IMM(30+L2_PREFETCH_DIST)) //Check if we have over 40 operations to do.
    VMOVAPS(ZMM(25), ZMM(1))    MOV(R(8), IMM(30))
    VMOVAPS(ZMM(26), ZMM(1))    MOV(R(9), IMM(8*8)) //amount to increment b* by each iteration
    VMOVAPS(ZMM(27), ZMM(1))    MOV(R(12), IMM(8*32)) //amount to increment a* by each iteration
    VMOVAPS(ZMM(28), ZMM(1))
    VMOVAPS(ZMM(29), ZMM(1))
    VMOVAPS(ZMM(30), ZMM(1))

#ifdef MONITORS
    RDTSC
    MOV(VAR(midl), EAX)
    MOV(VAR(midh), EDX)
#endif

    JLE(CONSIDER_UNDER_40)
    SUB(RSI, IMM(30+L2_PREFETCH_DIST))

    //First 30 iterations
    LABEL(LOOPREFECHCL2)
    MAIN_LOOP_PC_L2(R(8))
    JNZ(LOOPREFECHCL2)
    MOV(RCX, VAR(c))

    //Main Loop.
    LABEL(LOOPMAIN)
    MAIN_LOOP(RSI)
    JNZ(LOOPMAIN)

    //Penultimate 22 iterations.
    //Break these off from the main loop to avoid prefetching extra shit.
    MOV(R(14), VAR(a_next))
    MOV(R(13), VAR(b_next))
    SUB(R(14), R(15))
    SUB(R(13), RBX)
    //Yes, I know 10-20 = -10
    MOV(RSI, IMM(10+L2_PREFETCH_DIST-20))

    LABEL(LOOPMAIN2)
    MAIN_LOOP(RSI)
    JNZ(LOOPMAIN2)

    //Last 10 iterations
    MOV(R(8), IMM(10))

    LABEL(LOOPREFETCHCL1)
    MAIN_LOOP_PC_L1(R(8))
    JNZ(LOOPREFETCHCL1)

    JMP(POSTACCUM)

    //Alternate main loop, with no prefetching of C
    //Used when <= 40 iterations
    LABEL(CONSIDER_UNDER_40)

    MOV(RSI, VAR(k))
    TEST(RSI, RSI)
    JZ(POSTACCUM)

    LABEL(LOOP_UNDER_40)
    MAIN_LOOP(RSI)
    JNZ(LOOP_UNDER_40)

    LABEL(POSTACCUM)

#ifdef MONITORS
    RDTSC
    MOV(VAR(mid2l), EAX)
    MOV(VAR(mid2h), EDX)
#endif

    MOV(R(9), VAR(c)) //load address of c for update
    MOV(R(12), VAR(alpha)) //load address of alpha

    // Check if C is row stride. If not, jump to the slow scattered update
    MOV(R(14), VAR(cs_c))
    DEC(R(14))
    JNZ(SCATTEREDUPDATE)

    MOV(R(14), VAR(beta))
    VBROADCASTSD(ZMM(31), MEM(R(14)))

    MOV(RBX, MEM(R(14)))
    TEST(RBX, RBX)
    JZ(COLSTORBZ)

    UPDATE_C_4_ROWS( 1, 2, 3, 4)
    UPDATE_C_4_ROWS( 5, 6, 7, 8)
    UPDATE_C_4_ROWS( 9,10,11,12)
    UPDATE_C_4_ROWS(13,14,15,16)
    UPDATE_C_4_ROWS(17,18,19,20)
    UPDATE_C_4_ROWS(21,22,23,24)
    UPDATE_C_4_ROWS(25,26,27,28)
    UPDATE_C_2_ROWS(29,30)

    JMP(END)

    LABEL(COLSTORBZ)

    UPDATE_C_BZ_4_ROWS( 1, 2, 3, 4)
    UPDATE_C_BZ_4_ROWS( 5, 6, 7, 8)
    UPDATE_C_BZ_4_ROWS( 9,10,11,12)
    UPDATE_C_BZ_4_ROWS(13,14,15,16)
    UPDATE_C_BZ_4_ROWS(17,18,19,20)
    UPDATE_C_BZ_4_ROWS(21,22,23,24)
    UPDATE_C_BZ_4_ROWS(25,26,27,28)
    UPDATE_C_BZ_2_ROWS(29,30)

    JMP(END)

    LABEL(SCATTEREDUPDATE)

    MOV(R(13), VAR(beta))
    MOV(R(10), VAR(offsetPtr))
    VMOVAPS(ZMM(0), MEM(R(10)))
    MOV(RBX, MEM(R(13)))
    /* Note that this ignores the upper 32 bits in cs_c */
    VPBROADCASTD(ZMM(31), VAR(cs_c))
    VPMULLD(ZMM(0), ZMM(31), ZMM(0))

    TEST(RBX, RBX)
    JZ(SCATTERBZ)

    UPDATE_C_ROW_SCATTERED( 1)
    UPDATE_C_ROW_SCATTERED( 2)
    UPDATE_C_ROW_SCATTERED( 3)
    UPDATE_C_ROW_SCATTERED( 4)
    UPDATE_C_ROW_SCATTERED( 5)
    UPDATE_C_ROW_SCATTERED( 6)
    UPDATE_C_ROW_SCATTERED( 7)
    UPDATE_C_ROW_SCATTERED( 8)
    UPDATE_C_ROW_SCATTERED( 9)
    UPDATE_C_ROW_SCATTERED(10)
    UPDATE_C_ROW_SCATTERED(11)
    UPDATE_C_ROW_SCATTERED(12)
    UPDATE_C_ROW_SCATTERED(13)
    UPDATE_C_ROW_SCATTERED(14)
    UPDATE_C_ROW_SCATTERED(15)
    UPDATE_C_ROW_SCATTERED(16)
    UPDATE_C_ROW_SCATTERED(17)
    UPDATE_C_ROW_SCATTERED(18)
    UPDATE_C_ROW_SCATTERED(19)
    UPDATE_C_ROW_SCATTERED(20)
    UPDATE_C_ROW_SCATTERED(21)
    UPDATE_C_ROW_SCATTERED(22)
    UPDATE_C_ROW_SCATTERED(23)
    UPDATE_C_ROW_SCATTERED(24)
    UPDATE_C_ROW_SCATTERED(25)
    UPDATE_C_ROW_SCATTERED(26)
    UPDATE_C_ROW_SCATTERED(27)
    UPDATE_C_ROW_SCATTERED(28)
    UPDATE_C_ROW_SCATTERED(29)
    UPDATE_C_ROW_SCATTERED(30)

    JMP(END)

    LABEL(SCATTERBZ)

    UPDATE_C_BZ_ROW_SCATTERED( 1)
    UPDATE_C_BZ_ROW_SCATTERED( 2)
    UPDATE_C_BZ_ROW_SCATTERED( 3)
    UPDATE_C_BZ_ROW_SCATTERED( 4)
    UPDATE_C_BZ_ROW_SCATTERED( 5)
    UPDATE_C_BZ_ROW_SCATTERED( 6)
    UPDATE_C_BZ_ROW_SCATTERED( 7)
    UPDATE_C_BZ_ROW_SCATTERED( 8)
    UPDATE_C_BZ_ROW_SCATTERED( 9)
    UPDATE_C_BZ_ROW_SCATTERED(10)
    UPDATE_C_BZ_ROW_SCATTERED(11)
    UPDATE_C_BZ_ROW_SCATTERED(12)
    UPDATE_C_BZ_ROW_SCATTERED(13)
    UPDATE_C_BZ_ROW_SCATTERED(14)
    UPDATE_C_BZ_ROW_SCATTERED(15)
    UPDATE_C_BZ_ROW_SCATTERED(16)
    UPDATE_C_BZ_ROW_SCATTERED(17)
    UPDATE_C_BZ_ROW_SCATTERED(18)
    UPDATE_C_BZ_ROW_SCATTERED(19)
    UPDATE_C_BZ_ROW_SCATTERED(20)
    UPDATE_C_BZ_ROW_SCATTERED(21)
    UPDATE_C_BZ_ROW_SCATTERED(22)
    UPDATE_C_BZ_ROW_SCATTERED(23)
    UPDATE_C_BZ_ROW_SCATTERED(24)
    UPDATE_C_BZ_ROW_SCATTERED(25)
    UPDATE_C_BZ_ROW_SCATTERED(26)
    UPDATE_C_BZ_ROW_SCATTERED(27)
    UPDATE_C_BZ_ROW_SCATTERED(28)
    UPDATE_C_BZ_ROW_SCATTERED(29)
    UPDATE_C_BZ_ROW_SCATTERED(30)

    LABEL(END)

#ifdef MONITORS
    RDTSC
    MOV(VAR(botl), EAX)
    MOV(VAR(both), EDX)
#endif
    : // output operands
#ifdef MONITORS
      [topl]  "=m" (topl),
      [toph]  "=m" (toph),
      [midl]  "=m" (midl),
      [midh]  "=m" (midh),
      [mid2l] "=m" (mid2l),
      [mid2h] "=m" (mid2h),
      [botl]  "=m" (botl),
      [both]  "=m" (both)
#endif
    : // input operands
      [k]         "m" (k),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c),
      [a_next]    "m" (a_next),
      [b_next]    "m" (b_next),
      [offsetPtr] "m" (offsetPtr)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12",
      "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    );

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}
