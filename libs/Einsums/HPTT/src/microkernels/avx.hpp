//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <immintrin.h>

#ifdef EINSUMS_DEBUG
#ifdef __AVX2__
#warning "DEBUG INFO: Compiling HPTT with AVX2 as the highest level of vectorization."
#else
#warning "DEBUG INFO: Compiling HPTT with AVX1 as the highest level of vectorization."
#endif
#endif

template <bool betaIsZero, bool conjA>
struct micro_kernel<double, betaIsZero, conjA> {
    static void execute(double const *A, size_t const lda, size_t const innerStrideA, double *B, size_t const ldb,
                        size_t const innerStrideB, double const alpha, double const beta) {

        __m256d const alpha_reg = _mm256_set1_pd(alpha);

        __m256d rowA0, rowA1, rowA2, rowA3;

        if (innerStrideA != 1) {
#ifdef __AVX2__
            __m256i const indicesA = _mm256_set_epi64x(3 * innerStrideA, 2 * innerStrideA, 1 * innerStrideA, 0 * innerStrideA);

            rowA0 = _mm256_i64gather_pd((A + 0 * lda), indicesA, sizeof(double));
            rowA1 = _mm256_i64gather_pd((A + 1 * lda), indicesA, sizeof(double));
            rowA2 = _mm256_i64gather_pd((A + 2 * lda), indicesA, sizeof(double));
            rowA3 = _mm256_i64gather_pd((A + 3 * lda), indicesA, sizeof(double));
#else
            rowA0 = _mm256_set_pd(A[0 * lda + 3 * innerStrideA], A[0 * lda + 2 * innerStrideA], A[0 * lda + 1 * innerStrideA],
                                  A[0 * lda + 0 * innerStrideA]);
            rowA1 = _mm256_set_pd(A[1 * lda + 3 * innerStrideA], A[1 * lda + 2 * innerStrideA], A[1 * lda + 1 * innerStrideA],
                                  A[1 * lda + 0 * innerStrideA]);
            rowA2 = _mm256_set_pd(A[2 * lda + 3 * innerStrideA], A[2 * lda + 2 * innerStrideA], A[2 * lda + 1 * innerStrideA],
                                  A[2 * lda + 0 * innerStrideA]);
            rowA3 = _mm256_set_pd(A[3 * lda + 3 * innerStrideA], A[3 * lda + 2 * innerStrideA], A[3 * lda + 1 * innerStrideA],
                                  A[3 * lda + 0 * innerStrideA]);
#endif
        } else {
            rowA0 = _mm256_loadu_pd(A + 0 * lda);
            rowA1 = _mm256_loadu_pd(A + 1 * lda);
            rowA2 = _mm256_loadu_pd(A + 2 * lda);
            rowA3 = _mm256_loadu_pd(A + 3 * lda);
        }

        // 4x4 transpose kernel.
        __m256d row0_iter1, row1_iter1, row2_iter1, row3_iter1;

        // Solve 2x2 blocks.
        row0_iter1 = _mm256_unpacklo_pd(rowA0, rowA1);
        row1_iter1 = _mm256_unpackhi_pd(rowA0, rowA1);
        row2_iter1 = _mm256_unpacklo_pd(rowA2, rowA3);
        row3_iter1 = _mm256_unpackhi_pd(rowA2, rowA3);

        // Solve 4x4 blocks.
        rowA0 = _mm256_permute2f128_pd(row0_iter1, row2_iter1, 0x20);
        rowA1 = _mm256_permute2f128_pd(row1_iter1, row3_iter1, 0x20);
        rowA2 = _mm256_permute2f128_pd(row0_iter1, row2_iter1, 0x31);
        rowA3 = _mm256_permute2f128_pd(row1_iter1, row3_iter1, 0x31);

        // Scale.
        rowA0 = _mm256_mul_pd(rowA0, alpha_reg);
        rowA1 = _mm256_mul_pd(rowA1, alpha_reg);
        rowA2 = _mm256_mul_pd(rowA2, alpha_reg);
        rowA3 = _mm256_mul_pd(rowA3, alpha_reg);

        // Assign to B.
        if constexpr (!betaIsZero) {
            __m256d const beta_reg = _mm256_set1_pd(beta);

            __m256d rowB0, rowB1, rowB2, rowB3;

            if (innerStrideB != 1) {
#ifdef __AVX2__
                __m256i const indicesB = _mm256_set_epi64x(3 * innerStrideB, 2 * innerStrideB, 1 * innerStrideB, 0 * innerStrideB);

                rowB0 = _mm256_i64gather_pd((B + 0 * ldb), indicesB, sizeof(double));
                rowB1 = _mm256_i64gather_pd((B + 1 * ldb), indicesB, sizeof(double));
                rowB2 = _mm256_i64gather_pd((B + 2 * ldb), indicesB, sizeof(double));
                rowB3 = _mm256_i64gather_pd((B + 3 * ldb), indicesB, sizeof(double));
#else
                rowB0 = _mm256_set_pd(B[0 * ldb + 3 * innerStrideB], B[0 * ldb + 2 * innerStrideB], B[0 * ldb + 1 * innerStrideB],
                                      B[0 * ldb + 0 * innerStrideB]);
                rowB1 = _mm256_set_pd(B[1 * ldb + 3 * innerStrideB], B[1 * ldb + 2 * innerStrideB], B[1 * ldb + 1 * innerStrideB],
                                      B[1 * ldb + 0 * innerStrideB]);
                rowB2 = _mm256_set_pd(B[2 * ldb + 3 * innerStrideB], B[2 * ldb + 2 * innerStrideB], B[2 * ldb + 1 * innerStrideB],
                                      B[2 * ldb + 0 * innerStrideB]);
                rowB3 = _mm256_set_pd(B[3 * ldb + 3 * innerStrideB], B[3 * ldb + 2 * innerStrideB], B[3 * ldb + 1 * innerStrideB],
                                      B[3 * ldb + 0 * innerStrideB]);
#endif
            } else {
                rowB0 = _mm256_loadu_pd(B + 0 * ldb);
                rowB1 = _mm256_loadu_pd(B + 1 * ldb);
                rowB2 = _mm256_loadu_pd(B + 2 * ldb);
                rowB3 = _mm256_loadu_pd(B + 3 * ldb);
            }

#ifdef __FMA__
            rowB0 = _mm256_fmadd_pd(beta_reg, rowB0, rowA0);
            rowB1 = _mm256_fmadd_pd(beta_reg, rowB1, rowA1);
            rowB2 = _mm256_fmadd_pd(beta_reg, rowB2, rowA2);
            rowB3 = _mm256_fmadd_pd(beta_reg, rowB3, rowA3);
#elif defined(__FMA4__)
            rowB0 = _mm256_macc_pd(beta_reg, rowB0, rowA0);
            rowB1 = _mm256_macc_pd(beta_reg, rowB1, rowA1);
            rowB2 = _mm256_macc_pd(beta_reg, rowB2, rowA2);
            rowB3 = _mm256_macc_pd(beta_reg, rowB3, rowA3);
#else
            rowB0 = _mm256_add_pd(rowA0, _mm256_mul_pd(beta_reg, rowB0));
            rowB1 = _mm256_add_pd(rowA1, _mm256_mul_pd(beta_reg, rowB1));
            rowB2 = _mm256_add_pd(rowA2, _mm256_mul_pd(beta_reg, rowB2));
            rowB3 = _mm256_add_pd(rowA3, _mm256_mul_pd(beta_reg, rowB3));
#endif

            if (innerStrideB != 1) {
                double hold[4];

                _mm256_storeu_pd(hold, rowB0);

                for (int i = 0; i < 4; i++) {
                    B[0 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_pd(hold, rowB1);

                for (int i = 0; i < 4; i++) {
                    B[1 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_pd(hold, rowB2);

                for (int i = 0; i < 4; i++) {
                    B[2 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_pd(hold, rowB3);

                for (int i = 0; i < 4; i++) {
                    B[3 * ldb + i * innerStrideB] = hold[i];
                }
            } else {
                _mm256_storeu_pd((B + 0 * ldb), rowB0);
                _mm256_storeu_pd((B + 1 * ldb), rowB1);
                _mm256_storeu_pd((B + 2 * ldb), rowB2);
                _mm256_storeu_pd((B + 3 * ldb), rowB3);
            }
        } else {
            if (innerStrideB != 1) {
                double hold[4];

                _mm256_storeu_pd(hold, rowA0);

                for (int i = 0; i < 4; i++) {
                    B[0 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_pd(hold, rowA1);

                for (int i = 0; i < 4; i++) {
                    B[1 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_pd(hold, rowA2);

                for (int i = 0; i < 4; i++) {
                    B[2 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_pd(hold, rowA3);

                for (int i = 0; i < 4; i++) {
                    B[3 * ldb + i * innerStrideB] = hold[i];
                }
            } else {
                _mm256_storeu_pd((B + 0 * ldb), rowA0);
                _mm256_storeu_pd((B + 1 * ldb), rowA1);
                _mm256_storeu_pd((B + 2 * ldb), rowA2);
                _mm256_storeu_pd((B + 3 * ldb), rowA3);
            }
        }
    }
};

template <bool betaIsZero, bool conjA>
struct micro_kernel<float, betaIsZero, conjA> {
    static void execute(float const *A, size_t const lda, size_t const innerStrideA, float *B, size_t const ldb, size_t const innerStrideB,
                        float const alpha, float const beta) {

        __m256 alpha_reg = _mm256_set1_ps(alpha);

        __m256 rowA0, rowA1, rowA2, rowA3, rowA4, rowA5, rowA6, rowA7;

        if (innerStrideA != 1) {
#ifdef __AVX2__
            __m256i indicesA = _mm256_set_epi32(7 * innerStrideA, 6 * innerStrideA, 5 * innerStrideA, 4 * innerStrideA, 3 * innerStrideA,
                                                2 * innerStrideA, 1 * innerStrideA, 0 * innerStrideA);
            rowA0            = _mm256_i32gather_ps((A + 0 * lda), indicesA, sizeof(float));
            rowA1            = _mm256_i32gather_ps((A + 1 * lda), indicesA, sizeof(float));
            rowA2            = _mm256_i32gather_ps((A + 2 * lda), indicesA, sizeof(float));
            rowA3            = _mm256_i32gather_ps((A + 3 * lda), indicesA, sizeof(float));
            rowA4            = _mm256_i32gather_ps((A + 4 * lda), indicesA, sizeof(float));
            rowA5            = _mm256_i32gather_ps((A + 5 * lda), indicesA, sizeof(float));
            rowA6            = _mm256_i32gather_ps((A + 6 * lda), indicesA, sizeof(float));
            rowA7            = _mm256_i32gather_ps((A + 7 * lda), indicesA, sizeof(float));
#else
            rowA0 = _mm256_set_ps(A[0 * lda + 7 * innerStrideA], A[0 * lda + 6 * innerStrideA], A[0 * lda + 5 * innerStrideA],
                                  A[0 * lda + 4 * innerStrideA], A[0 * lda + 3 * innerStrideA], A[0 * lda + 2 * innerStrideA],
                                  A[0 * lda + 1 * innerStrideA], A[0 * lda + 0 * innerStrideA]);
            rowA1 = _mm256_set_ps(A[1 * lda + 7 * innerStrideA], A[1 * lda + 6 * innerStrideA], A[1 * lda + 5 * innerStrideA],
                                  A[1 * lda + 4 * innerStrideA], A[1 * lda + 3 * innerStrideA], A[1 * lda + 2 * innerStrideA],
                                  A[1 * lda + 1 * innerStrideA], A[1 * lda + 0 * innerStrideA]);
            rowA2 = _mm256_set_ps(A[2 * lda + 7 * innerStrideA], A[2 * lda + 6 * innerStrideA], A[2 * lda + 5 * innerStrideA],
                                  A[2 * lda + 4 * innerStrideA], A[2 * lda + 3 * innerStrideA], A[2 * lda + 2 * innerStrideA],
                                  A[2 * lda + 1 * innerStrideA], A[2 * lda + 0 * innerStrideA]);
            rowA3 = _mm256_set_ps(A[3 * lda + 7 * innerStrideA], A[3 * lda + 6 * innerStrideA], A[3 * lda + 5 * innerStrideA],
                                  A[3 * lda + 4 * innerStrideA], A[3 * lda + 3 * innerStrideA], A[3 * lda + 2 * innerStrideA],
                                  A[3 * lda + 1 * innerStrideA], A[3 * lda + 0 * innerStrideA]);
            rowA4 = _mm256_set_ps(A[4 * lda + 7 * innerStrideA], A[4 * lda + 6 * innerStrideA], A[4 * lda + 5 * innerStrideA],
                                  A[4 * lda + 4 * innerStrideA], A[4 * lda + 3 * innerStrideA], A[4 * lda + 2 * innerStrideA],
                                  A[4 * lda + 1 * innerStrideA], A[4 * lda + 0 * innerStrideA]);
            rowA5 = _mm256_set_ps(A[5 * lda + 7 * innerStrideA], A[5 * lda + 6 * innerStrideA], A[5 * lda + 5 * innerStrideA],
                                  A[5 * lda + 4 * innerStrideA], A[5 * lda + 3 * innerStrideA], A[5 * lda + 2 * innerStrideA],
                                  A[5 * lda + 1 * innerStrideA], A[5 * lda + 0 * innerStrideA]);
            rowA6 = _mm256_set_ps(A[6 * lda + 7 * innerStrideA], A[6 * lda + 6 * innerStrideA], A[6 * lda + 5 * innerStrideA],
                                  A[6 * lda + 4 * innerStrideA], A[6 * lda + 3 * innerStrideA], A[6 * lda + 2 * innerStrideA],
                                  A[6 * lda + 1 * innerStrideA], A[6 * lda + 0 * innerStrideA]);
            rowA7 = _mm256_set_ps(A[7 * lda + 7 * innerStrideA], A[7 * lda + 6 * innerStrideA], A[7 * lda + 5 * innerStrideA],
                                  A[7 * lda + 4 * innerStrideA], A[7 * lda + 3 * innerStrideA], A[7 * lda + 2 * innerStrideA],
                                  A[7 * lda + 1 * innerStrideA], A[7 * lda + 0 * innerStrideA]);
#endif
        } else {
            rowA0 = _mm256_loadu_ps((A + 0 * lda));
            rowA1 = _mm256_loadu_ps((A + 1 * lda));
            rowA2 = _mm256_loadu_ps((A + 2 * lda));
            rowA3 = _mm256_loadu_ps((A + 3 * lda));
            rowA4 = _mm256_loadu_ps((A + 4 * lda));
            rowA5 = _mm256_loadu_ps((A + 5 * lda));
            rowA6 = _mm256_loadu_ps((A + 6 * lda));
            rowA7 = _mm256_loadu_ps((A + 7 * lda));
        }

        // 8x8 transpose kernel.
        
        // Solve the 2x2 diagonal blocks.
        // unpack doesn't work the same for 32-bit floats. Re-figure the transpose pattern
        // Set up for the solve
        __m256 const row0_iter1 = _mm256_unpacklo_ps(rowA0, rowA1);
        __m256 const row1_iter1 = _mm256_unpackhi_ps(rowA0, rowA1);
        __m256 const row2_iter1 = _mm256_unpacklo_ps(rowA2, rowA3);
        __m256 const row3_iter1 = _mm256_unpackhi_ps(rowA2, rowA3);
        __m256 const row4_iter1 = _mm256_unpacklo_ps(rowA4, rowA5);
        __m256 const row5_iter1 = _mm256_unpackhi_ps(rowA4, rowA5);
        __m256 const row6_iter1 = _mm256_unpacklo_ps(rowA6, rowA7);
        __m256 const row7_iter1 = _mm256_unpackhi_ps(rowA6, rowA7);

        // Finish 2x2 blocks.
        __m256 const row0_iter2 = _mm256_shuffle_ps(row0_iter1, row1_iter1, 0x44);
        __m256 const row1_iter2 = _mm256_shuffle_ps(row0_iter1, row1_iter1, 0xee);
        __m256 const row2_iter2 = _mm256_shuffle_ps(row2_iter1, row3_iter1, 0x44);
        __m256 const row3_iter2 = _mm256_shuffle_ps(row2_iter1, row3_iter1, 0xee);
        __m256 const row4_iter2 = _mm256_shuffle_ps(row4_iter1, row5_iter1, 0x44);
        __m256 const row5_iter2 = _mm256_shuffle_ps(row4_iter1, row5_iter1, 0xee);
        __m256 const row6_iter2 = _mm256_shuffle_ps(row6_iter1, row7_iter1, 0x44);
        __m256 const row7_iter2 = _mm256_shuffle_ps(row6_iter1, row7_iter1, 0xee);

        // Solve the 4x4 diagonal blocks.
        __m256 const row0_iter3 = _mm256_shuffle_ps(row0_iter2, row2_iter2, 0x44);
        __m256 const row1_iter3 = _mm256_shuffle_ps(row1_iter2, row3_iter2, 0x44);
        __m256 const row2_iter3 = _mm256_shuffle_ps(row0_iter2, row2_iter2, 0xee);
        __m256 const row3_iter3 = _mm256_shuffle_ps(row1_iter2, row3_iter2, 0xee);
        __m256 const row4_iter3 = _mm256_shuffle_ps(row4_iter2, row6_iter2, 0x44);
        __m256 const row5_iter3 = _mm256_shuffle_ps(row5_iter2, row7_iter2, 0x44);
        __m256 const row6_iter3 = _mm256_shuffle_ps(row4_iter2, row6_iter2, 0xee);
        __m256 const row7_iter3 = _mm256_shuffle_ps(row5_iter2, row7_iter2, 0xee);

        // Solve the rest.
        rowA0 = _mm256_permute2f128_ps(row0_iter3, row4_iter3, 0x20);
        rowA1 = _mm256_permute2f128_ps(row1_iter3, row5_iter3, 0x20);
        rowA2 = _mm256_permute2f128_ps(row2_iter3, row6_iter3, 0x20);
        rowA3 = _mm256_permute2f128_ps(row3_iter3, row7_iter3, 0x20);
        rowA4 = _mm256_permute2f128_ps(row0_iter3, row4_iter3, 0x31);
        rowA5 = _mm256_permute2f128_ps(row1_iter3, row5_iter3, 0x31);
        rowA6 = _mm256_permute2f128_ps(row2_iter3, row6_iter3, 0x31);
        rowA7 = _mm256_permute2f128_ps(row3_iter3, row7_iter3, 0x31);

        // Scale A
        rowA0 = _mm256_mul_ps(rowA0, alpha_reg);
        rowA1 = _mm256_mul_ps(rowA1, alpha_reg);
        rowA2 = _mm256_mul_ps(rowA2, alpha_reg);
        rowA3 = _mm256_mul_ps(rowA3, alpha_reg);
        rowA4 = _mm256_mul_ps(rowA4, alpha_reg);
        rowA5 = _mm256_mul_ps(rowA5, alpha_reg);
        rowA6 = _mm256_mul_ps(rowA6, alpha_reg);
        rowA7 = _mm256_mul_ps(rowA7, alpha_reg);

        // Store into B.
        if constexpr (!betaIsZero) {
            __m256 beta_reg = _mm256_set1_ps(beta);

            __m256 rowB0, rowB1, rowB2, rowB3, rowB4, rowB5, rowB6, rowB7;

            if (innerStrideB != 1) {
#ifdef __AVX2__
                __m256i indicesB = _mm256_set_epi32(7 * innerStrideB, 6 * innerStrideB, 5 * innerStrideB, 4 * innerStrideB,
                                                    3 * innerStrideB, 2 * innerStrideB, 1 * innerStrideB, 0 * innerStrideB);
                rowB0            = _mm256_i32gather_ps((B + 0 * ldb), indicesB, sizeof(float));
                rowB1            = _mm256_i32gather_ps((B + 1 * ldb), indicesB, sizeof(float));
                rowB2            = _mm256_i32gather_ps((B + 2 * ldb), indicesB, sizeof(float));
                rowB3            = _mm256_i32gather_ps((B + 3 * ldb), indicesB, sizeof(float));
                rowB4            = _mm256_i32gather_ps((B + 4 * ldb), indicesB, sizeof(float));
                rowB5            = _mm256_i32gather_ps((B + 5 * ldb), indicesB, sizeof(float));
                rowB6            = _mm256_i32gather_ps((B + 6 * ldb), indicesB, sizeof(float));
                rowB7            = _mm256_i32gather_ps((B + 7 * ldb), indicesB, sizeof(float));
#else
                rowB0 = _mm256_set_ps(B[0 * ldb + 7 * innerStrideB], B[0 * ldb + 6 * innerStrideB], B[0 * ldb + 5 * innerStrideB],
                                      B[0 * ldb + 4 * innerStrideB], B[0 * ldb + 3 * innerStrideB], B[0 * ldb + 2 * innerStrideB],
                                      B[0 * ldb + 1 * innerStrideB], B[0 * ldb + 0 * innerStrideB]);
                rowB1 = _mm256_set_ps(B[1 * ldb + 7 * innerStrideB], B[1 * ldb + 6 * innerStrideB], B[1 * ldb + 5 * innerStrideB],
                                      B[1 * ldb + 4 * innerStrideB], B[1 * ldb + 3 * innerStrideB], B[1 * ldb + 2 * innerStrideB],
                                      B[1 * ldb + 1 * innerStrideB], B[1 * ldb + 0 * innerStrideB]);
                rowB2 = _mm256_set_ps(B[2 * ldb + 7 * innerStrideB], B[2 * ldb + 6 * innerStrideB], B[2 * ldb + 5 * innerStrideB],
                                      B[2 * ldb + 4 * innerStrideB], B[2 * ldb + 3 * innerStrideB], B[2 * ldb + 2 * innerStrideB],
                                      B[2 * ldb + 1 * innerStrideB], B[2 * ldb + 0 * innerStrideB]);
                rowB3 = _mm256_set_ps(B[3 * ldb + 7 * innerStrideB], B[3 * ldb + 6 * innerStrideB], B[3 * ldb + 5 * innerStrideB],
                                      B[3 * ldb + 4 * innerStrideB], B[3 * ldb + 3 * innerStrideB], B[3 * ldb + 2 * innerStrideB],
                                      B[3 * ldb + 1 * innerStrideB], B[3 * ldb + 0 * innerStrideB]);
                rowB4 = _mm256_set_ps(B[4 * ldb + 7 * innerStrideB], B[4 * ldb + 6 * innerStrideB], B[4 * ldb + 5 * innerStrideB],
                                      B[4 * ldb + 4 * innerStrideB], B[4 * ldb + 3 * innerStrideB], B[4 * ldb + 2 * innerStrideB],
                                      B[4 * ldb + 1 * innerStrideB], B[4 * ldb + 0 * innerStrideB]);
                rowB5 = _mm256_set_ps(B[5 * ldb + 7 * innerStrideB], B[5 * ldb + 6 * innerStrideB], B[5 * ldb + 5 * innerStrideB],
                                      B[5 * ldb + 4 * innerStrideB], B[5 * ldb + 3 * innerStrideB], B[5 * ldb + 2 * innerStrideB],
                                      B[5 * ldb + 1 * innerStrideB], B[5 * ldb + 0 * innerStrideB]);
                rowB6 = _mm256_set_ps(B[6 * ldb + 7 * innerStrideB], B[6 * ldb + 6 * innerStrideB], B[6 * ldb + 5 * innerStrideB],
                                      B[6 * ldb + 4 * innerStrideB], B[6 * ldb + 3 * innerStrideB], B[6 * ldb + 2 * innerStrideB],
                                      B[6 * ldb + 1 * innerStrideB], B[6 * ldb + 0 * innerStrideB]);
                rowB7 = _mm256_set_ps(B[7 * ldb + 7 * innerStrideB], B[7 * ldb + 6 * innerStrideB], B[7 * ldb + 5 * innerStrideB],
                                      B[7 * ldb + 4 * innerStrideB], B[7 * ldb + 3 * innerStrideB], B[7 * ldb + 2 * innerStrideB],
                                      B[7 * ldb + 1 * innerStrideB], B[7 * ldb + 0 * innerStrideB]);
#endif
            } else {
                rowB0 = _mm256_loadu_ps((B + 0 * ldb));
                rowB1 = _mm256_loadu_ps((B + 1 * ldb));
                rowB2 = _mm256_loadu_ps((B + 2 * ldb));
                rowB3 = _mm256_loadu_ps((B + 3 * ldb));
                rowB4 = _mm256_loadu_ps((B + 4 * ldb));
                rowB5 = _mm256_loadu_ps((B + 5 * ldb));
                rowB6 = _mm256_loadu_ps((B + 6 * ldb));
                rowB7 = _mm256_loadu_ps((B + 7 * ldb));
            }

#ifdef __FMA__
            rowB0 = _mm256_fmadd_ps(beta_reg, rowB0, rowA0);
            rowB1 = _mm256_fmadd_ps(beta_reg, rowB1, rowA1);
            rowB2 = _mm256_fmadd_ps(beta_reg, rowB2, rowA2);
            rowB3 = _mm256_fmadd_ps(beta_reg, rowB3, rowA3);
            rowB4 = _mm256_fmadd_ps(beta_reg, rowB4, rowA4);
            rowB5 = _mm256_fmadd_ps(beta_reg, rowB5, rowA5);
            rowB6 = _mm256_fmadd_ps(beta_reg, rowB6, rowA6);
            rowB7 = _mm256_fmadd_ps(beta_reg, rowB7, rowA7);
#elif defined(__FMA4__)
            rowB0 = _mm256_macc_ps(beta_reg, rowB0, rowA0);
            rowB1 = _mm256_macc_ps(beta_reg, rowB1, rowA1);
            rowB2 = _mm256_macc_ps(beta_reg, rowB2, rowA2);
            rowB3 = _mm256_macc_ps(beta_reg, rowB3, rowA3);
            rowB4 = _mm256_macc_ps(beta_reg, rowB4, rowA4);
            rowB5 = _mm256_macc_ps(beta_reg, rowB5, rowA5);
            rowB6 = _mm256_macc_ps(beta_reg, rowB6, rowA6);
            rowB7 = _mm256_macc_ps(beta_reg, rowB7, rowA7);
#else
            rowB0 = _mm256_add_ps(rowA0, _mm256_mul_ps(beta_reg, rowB0));
            rowB1 = _mm256_add_ps(rowA1, _mm256_mul_ps(beta_reg, rowB1));
            rowB2 = _mm256_add_ps(rowA2, _mm256_mul_ps(beta_reg, rowB2));
            rowB3 = _mm256_add_ps(rowA3, _mm256_mul_ps(beta_reg, rowB3));
            rowB4 = _mm256_add_ps(rowA4, _mm256_mul_ps(beta_reg, rowB4));
            rowB5 = _mm256_add_ps(rowA5, _mm256_mul_ps(beta_reg, rowB5));
            rowB6 = _mm256_add_ps(rowA6, _mm256_mul_ps(beta_reg, rowB6));
            rowB7 = _mm256_add_ps(rowA7, _mm256_mul_ps(beta_reg, rowB7));
#endif

            if (innerStrideB != 1) {
                float hold[8];

                _mm256_storeu_ps(hold, rowB0);

                for (int i = 0; i < 8; i++) {
                    B[0 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB1);

                for (int i = 0; i < 8; i++) {
                    B[1 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB2);

                for (int i = 0; i < 8; i++) {
                    B[2 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB3);

                for (int i = 0; i < 8; i++) {
                    B[3 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB4);

                for (int i = 0; i < 8; i++) {
                    B[4 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB5);

                for (int i = 0; i < 8; i++) {
                    B[5 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB6);

                for (int i = 0; i < 8; i++) {
                    B[6 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowB7);

                for (int i = 0; i < 8; i++) {
                    B[7 * ldb + i * innerStrideB] = hold[i];
                }

            } else {
                _mm256_storeu_ps((B + 0 * ldb), rowB0);
                _mm256_storeu_ps((B + 1 * ldb), rowB1);
                _mm256_storeu_ps((B + 2 * ldb), rowB2);
                _mm256_storeu_ps((B + 3 * ldb), rowB3);
                _mm256_storeu_ps((B + 4 * ldb), rowB4);
                _mm256_storeu_ps((B + 5 * ldb), rowB5);
                _mm256_storeu_ps((B + 6 * ldb), rowB6);
                _mm256_storeu_ps((B + 7 * ldb), rowB7);
            }
        } else {
            if (innerStrideB != 1) {
                float hold[8];

                _mm256_storeu_ps(hold, rowA0);

                for (int i = 0; i < 8; i++) {
                    B[0 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA1);

                for (int i = 0; i < 8; i++) {
                    B[1 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA2);

                for (int i = 0; i < 8; i++) {
                    B[2 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA3);

                for (int i = 0; i < 8; i++) {
                    B[3 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA4);

                for (int i = 0; i < 8; i++) {
                    B[4 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA5);

                for (int i = 0; i < 8; i++) {
                    B[5 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA6);

                for (int i = 0; i < 8; i++) {
                    B[6 * ldb + i * innerStrideB] = hold[i];
                }

                _mm256_storeu_ps(hold, rowA7);

                for (int i = 0; i < 8; i++) {
                    B[7 * ldb + i * innerStrideB] = hold[i];
                }

            } else {
                _mm256_storeu_ps((B + 0 * ldb), rowA0);
                _mm256_storeu_ps((B + 1 * ldb), rowA1);
                _mm256_storeu_ps((B + 2 * ldb), rowA2);
                _mm256_storeu_ps((B + 3 * ldb), rowA3);
                _mm256_storeu_ps((B + 4 * ldb), rowA4);
                _mm256_storeu_ps((B + 5 * ldb), rowA5);
                _mm256_storeu_ps((B + 6 * ldb), rowA6);
                _mm256_storeu_ps((B + 7 * ldb), rowA7);
            }
        }
    }
};

template <>
void streamingStore<float>(float *out, float const *in) {
    _mm256_stream_ps(out, _mm256_loadu_ps(in));
}
template <>
void streamingStore<double>(double *out, double const *in) {
    _mm256_stream_pd(out, _mm256_loadu_pd(in));
}