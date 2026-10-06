//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------
#include <immintrin.h>

#ifdef EINSUMS_DEBUG
#    warning "DEBUG INFO: Compiling HPTT with AVX512 as the highest level of vectorization."
#endif

template <bool betaIsZero, bool conjA>
struct micro_kernel<double, betaIsZero, conjA> {
    static void execute(double const *A, size_t const lda, size_t const innerStrideA, double *B, size_t const ldb,
                        size_t const innerStrideB, double const alpha, double const beta) {
        __m512d const reg_alpha = _mm512_set1_pd(alpha); // do not alter the content of B
                                                         // Load A
        __m512d rowA0, rowA1, rowA2, rowA3, rowA4, rowA5, rowA6, rowA7;

        if (innerStrideA != 1) {
            __m512i const indicesA = _mm512_set_epi64(7 * innerStrideA, 6 * innerStrideA, 5 * innerStrideA, 4 * innerStrideA,
                                                      3 * innerStrideA, 2 * innerStrideA, 1 * innerStrideA, 0 * innerStrideA);
            rowA0 = _mm512_i64gather_pd(indicesA, (A + 0 * lda), sizeof(double)); // The mm256 and mm512 versions flip their arguments.
            rowA1 = _mm512_i64gather_pd(indicesA, (A + 1 * lda), sizeof(double));
            rowA2 = _mm512_i64gather_pd(indicesA, (A + 2 * lda), sizeof(double));
            rowA3 = _mm512_i64gather_pd(indicesA, (A + 3 * lda), sizeof(double));
            rowA4 = _mm512_i64gather_pd(indicesA, (A + 4 * lda), sizeof(double));
            rowA5 = _mm512_i64gather_pd(indicesA, (A + 5 * lda), sizeof(double));
            rowA6 = _mm512_i64gather_pd(indicesA, (A + 6 * lda), sizeof(double));
            rowA7 = _mm512_i64gather_pd(indicesA, (A + 7 * lda), sizeof(double));
        } else {
            rowA0 = _mm512_loadu_pd((A + 0 * lda));
            rowA1 = _mm512_loadu_pd((A + 1 * lda));
            rowA2 = _mm512_loadu_pd((A + 2 * lda));
            rowA3 = _mm512_loadu_pd((A + 3 * lda));
            rowA4 = _mm512_loadu_pd((A + 4 * lda));
            rowA5 = _mm512_loadu_pd((A + 5 * lda));
            rowA6 = _mm512_loadu_pd((A + 6 * lda));
            rowA7 = _mm512_loadu_pd((A + 7 * lda));
        }

        // 8x8 transpose micro kernel

        // Solve the 2x2 blocks on the diagonal.
        __m512d const row0_iter1 = _mm512_unpacklo_pd(rowA0, rowA1);
        __m512d const row1_iter1 = _mm512_unpackhi_pd(rowA0, rowA1);
        __m512d const row2_iter1 = _mm512_unpacklo_pd(rowA2, rowA3);
        __m512d const row3_iter1 = _mm512_unpackhi_pd(rowA2, rowA3);
        __m512d const row4_iter1 = _mm512_unpacklo_pd(rowA4, rowA5);
        __m512d const row5_iter1 = _mm512_unpackhi_pd(rowA4, rowA5);
        __m512d const row6_iter1 = _mm512_unpacklo_pd(rowA6, rowA7);
        __m512d const row7_iter1 = _mm512_unpackhi_pd(rowA6, rowA7);

        __m512d const row0_iter2 = _mm512_shuffle_f64x2(row0_iter1, row2_iter1, 0x88);
        __m512d const row1_iter2 = _mm512_shuffle_f64x2(row1_iter1, row3_iter1, 0x88);
        __m512d const row2_iter2 = _mm512_shuffle_f64x2(row0_iter1, row2_iter1, 0xdd);
        __m512d const row3_iter2 = _mm512_shuffle_f64x2(row1_iter1, row3_iter1, 0xdd);
        __m512d const row4_iter2 = _mm512_shuffle_f64x2(row4_iter1, row6_iter1, 0x88);
        __m512d const row5_iter2 = _mm512_shuffle_f64x2(row5_iter1, row7_iter1, 0x88);
        __m512d const row6_iter2 = _mm512_shuffle_f64x2(row4_iter1, row6_iter1, 0xdd);
        __m512d const row7_iter2 = _mm512_shuffle_f64x2(row5_iter1, row7_iter1, 0xdd);

        rowA0 = _mm512_shuffle_f64x2(row0_iter2, row4_iter2, 0x88);
        rowA1 = _mm512_shuffle_f64x2(row1_iter2, row5_iter2, 0x88);
        rowA2 = _mm512_shuffle_f64x2(row2_iter2, row6_iter2, 0x88);
        rowA3 = _mm512_shuffle_f64x2(row3_iter2, row7_iter2, 0x88);
        rowA4 = _mm512_shuffle_f64x2(row0_iter2, row4_iter2, 0xdd);
        rowA5 = _mm512_shuffle_f64x2(row1_iter2, row5_iter2, 0xdd);
        rowA6 = _mm512_shuffle_f64x2(row2_iter2, row6_iter2, 0xdd);
        rowA7 = _mm512_shuffle_f64x2(row3_iter2, row7_iter2, 0xdd);

        // Scale A
        rowA0 = _mm512_mul_pd(rowA0, reg_alpha);
        rowA1 = _mm512_mul_pd(rowA1, reg_alpha);
        rowA2 = _mm512_mul_pd(rowA2, reg_alpha);
        rowA3 = _mm512_mul_pd(rowA3, reg_alpha);
        rowA4 = _mm512_mul_pd(rowA4, reg_alpha);
        rowA5 = _mm512_mul_pd(rowA5, reg_alpha);
        rowA6 = _mm512_mul_pd(rowA6, reg_alpha);
        rowA7 = _mm512_mul_pd(rowA7, reg_alpha);

        // Load B
        if constexpr (!betaIsZero) {
            __m512d rowB0, rowB1, rowB2, rowB3, rowB4, rowB5, rowB6, rowB7;
            __m512i indicesB;
            if (innerStrideB != 1) {
                indicesB = _mm512_set_epi64(7 * innerStrideB, 6 * innerStrideB, 5 * innerStrideB, 4 * innerStrideB, 3 * innerStrideB,
                                            2 * innerStrideB, 1 * innerStrideB, 0 * innerStrideB);
                rowB0    = _mm512_i64gather_pd(indicesB, (B + 0 * ldb), sizeof(double));
                rowB1    = _mm512_i64gather_pd(indicesB, (B + 1 * ldb), sizeof(double));
                rowB2    = _mm512_i64gather_pd(indicesB, (B + 2 * ldb), sizeof(double));
                rowB3    = _mm512_i64gather_pd(indicesB, (B + 3 * ldb), sizeof(double));
                rowB4    = _mm512_i64gather_pd(indicesB, (B + 4 * ldb), sizeof(double));
                rowB5    = _mm512_i64gather_pd(indicesB, (B + 5 * ldb), sizeof(double));
                rowB6    = _mm512_i64gather_pd(indicesB, (B + 6 * ldb), sizeof(double));
                rowB7    = _mm512_i64gather_pd(indicesB, (B + 7 * ldb), sizeof(double));
            } else {
                rowB0 = _mm512_loadu_pd((B + 0 * ldb));
                rowB1 = _mm512_loadu_pd((B + 1 * ldb));
                rowB2 = _mm512_loadu_pd((B + 2 * ldb));
                rowB3 = _mm512_loadu_pd((B + 3 * ldb));
                rowB4 = _mm512_loadu_pd((B + 4 * ldb));
                rowB5 = _mm512_loadu_pd((B + 5 * ldb));
                rowB6 = _mm512_loadu_pd((B + 6 * ldb));
                rowB7 = _mm512_loadu_pd((B + 7 * ldb));
            }

            __m512d const reg_beta = _mm512_set1_pd(beta);

            rowB0 = _mm512_fmadd_pd(rowB0, reg_beta, rowA0);
            rowB1 = _mm512_fmadd_pd(rowB1, reg_beta, rowA1);
            rowB2 = _mm512_fmadd_pd(rowB2, reg_beta, rowA2);
            rowB3 = _mm512_fmadd_pd(rowB3, reg_beta, rowA3);
            rowB4 = _mm512_fmadd_pd(rowB4, reg_beta, rowA4);
            rowB5 = _mm512_fmadd_pd(rowB5, reg_beta, rowA5);
            rowB6 = _mm512_fmadd_pd(rowB6, reg_beta, rowA6);
            rowB7 = _mm512_fmadd_pd(rowB7, reg_beta, rowA7);
            // Store B
            if (innerStrideB != 1) {
                _mm512_i64scatter_pd((B + 0 * ldb), indicesB, rowB0, sizeof(double));
                _mm512_i64scatter_pd((B + 1 * ldb), indicesB, rowB1, sizeof(double));
                _mm512_i64scatter_pd((B + 2 * ldb), indicesB, rowB2, sizeof(double));
                _mm512_i64scatter_pd((B + 3 * ldb), indicesB, rowB3, sizeof(double));
                _mm512_i64scatter_pd((B + 4 * ldb), indicesB, rowB4, sizeof(double));
                _mm512_i64scatter_pd((B + 5 * ldb), indicesB, rowB5, sizeof(double));
                _mm512_i64scatter_pd((B + 6 * ldb), indicesB, rowB6, sizeof(double));
                _mm512_i64scatter_pd((B + 7 * ldb), indicesB, rowB7, sizeof(double));
            } else {
                _mm512_storeu_pd((B + 0 * ldb), rowB0);
                _mm512_storeu_pd((B + 1 * ldb), rowB1);
                _mm512_storeu_pd((B + 2 * ldb), rowB2);
                _mm512_storeu_pd((B + 3 * ldb), rowB3);
                _mm512_storeu_pd((B + 4 * ldb), rowB4);
                _mm512_storeu_pd((B + 5 * ldb), rowB5);
                _mm512_storeu_pd((B + 6 * ldb), rowB6);
                _mm512_storeu_pd((B + 7 * ldb), rowB7);
            }
        } else {
            // Store B
            if (innerStrideB != 1) {
                __m512i const indicesB = _mm512_set_epi64(7 * innerStrideB, 6 * innerStrideB, 5 * innerStrideB, 4 * innerStrideB,
                                                          3 * innerStrideB, 2 * innerStrideB, 1 * innerStrideB, 0 * innerStrideB);
                _mm512_i64scatter_pd((B + 0 * ldb), indicesB, rowA0, sizeof(double));
                _mm512_i64scatter_pd((B + 1 * ldb), indicesB, rowA1, sizeof(double));
                _mm512_i64scatter_pd((B + 2 * ldb), indicesB, rowA2, sizeof(double));
                _mm512_i64scatter_pd((B + 3 * ldb), indicesB, rowA3, sizeof(double));
                _mm512_i64scatter_pd((B + 4 * ldb), indicesB, rowA4, sizeof(double));
                _mm512_i64scatter_pd((B + 5 * ldb), indicesB, rowA5, sizeof(double));
                _mm512_i64scatter_pd((B + 6 * ldb), indicesB, rowA6, sizeof(double));
                _mm512_i64scatter_pd((B + 7 * ldb), indicesB, rowA7, sizeof(double));
            } else {
                _mm512_storeu_pd((B + 0 * ldb), rowA0);
                _mm512_storeu_pd((B + 1 * ldb), rowA1);
                _mm512_storeu_pd((B + 2 * ldb), rowA2);
                _mm512_storeu_pd((B + 3 * ldb), rowA3);
                _mm512_storeu_pd((B + 4 * ldb), rowA4);
                _mm512_storeu_pd((B + 5 * ldb), rowA5);
                _mm512_storeu_pd((B + 6 * ldb), rowA6);
                _mm512_storeu_pd((B + 7 * ldb), rowA7);
            }
        }
    }
};

template <bool betaIsZero, bool conjA>
struct micro_kernel<float, betaIsZero, conjA> {
    static void execute(float const *A, size_t const lda, size_t const innerStrideA, float *B, size_t const ldb, size_t const innerStrideB,
                        float const alpha, float const beta) {
        __m512 const reg_alpha = _mm512_set1_ps(alpha); // do not alter the content of B
        // Load A
        __m512 rowA0, rowA1, rowA2, rowA3, rowA4, rowA5, rowA6, rowA7, rowA8, rowA9, rowA10, rowA11, rowA12, rowA13, rowA14, rowA15;
        if (innerStrideA == 1) {
            rowA0  = _mm512_loadu_ps((A + 0 * lda));
            rowA1  = _mm512_loadu_ps((A + 1 * lda));
            rowA2  = _mm512_loadu_ps((A + 2 * lda));
            rowA3  = _mm512_loadu_ps((A + 3 * lda));
            rowA4  = _mm512_loadu_ps((A + 4 * lda));
            rowA5  = _mm512_loadu_ps((A + 5 * lda));
            rowA6  = _mm512_loadu_ps((A + 6 * lda));
            rowA7  = _mm512_loadu_ps((A + 7 * lda));
            rowA8  = _mm512_loadu_ps((A + 8 * lda));
            rowA9  = _mm512_loadu_ps((A + 9 * lda));
            rowA10 = _mm512_loadu_ps((A + 10 * lda));
            rowA11 = _mm512_loadu_ps((A + 11 * lda));
            rowA12 = _mm512_loadu_ps((A + 12 * lda));
            rowA13 = _mm512_loadu_ps((A + 13 * lda));
            rowA14 = _mm512_loadu_ps((A + 14 * lda));
            rowA15 = _mm512_loadu_ps((A + 15 * lda));
        } else {
            __m512i const indicesA = _mm512_set_epi32(15 * innerStrideA, 14 * innerStrideA, 13 * innerStrideA, 12 * innerStrideA,
                                                      11 * innerStrideA, 10 * innerStrideA, 9 * innerStrideA, 8 * innerStrideA,
                                                      7 * innerStrideA, 6 * innerStrideA, 5 * innerStrideA, 4 * innerStrideA,
                                                      3 * innerStrideA, 2 * innerStrideA, 1 * innerStrideA, 0 * innerStrideA);

            rowA0  = _mm512_i32gather_ps(indicesA, (A + 0 * lda), sizeof(float));
            rowA1  = _mm512_i32gather_ps(indicesA, (A + 1 * lda), sizeof(float));
            rowA2  = _mm512_i32gather_ps(indicesA, (A + 2 * lda), sizeof(float));
            rowA3  = _mm512_i32gather_ps(indicesA, (A + 3 * lda), sizeof(float));
            rowA4  = _mm512_i32gather_ps(indicesA, (A + 4 * lda), sizeof(float));
            rowA5  = _mm512_i32gather_ps(indicesA, (A + 5 * lda), sizeof(float));
            rowA6  = _mm512_i32gather_ps(indicesA, (A + 6 * lda), sizeof(float));
            rowA7  = _mm512_i32gather_ps(indicesA, (A + 7 * lda), sizeof(float));
            rowA8  = _mm512_i32gather_ps(indicesA, (A + 8 * lda), sizeof(float));
            rowA9  = _mm512_i32gather_ps(indicesA, (A + 9 * lda), sizeof(float));
            rowA10 = _mm512_i32gather_ps(indicesA, (A + 10 * lda), sizeof(float));
            rowA11 = _mm512_i32gather_ps(indicesA, (A + 11 * lda), sizeof(float));
            rowA12 = _mm512_i32gather_ps(indicesA, (A + 12 * lda), sizeof(float));
            rowA13 = _mm512_i32gather_ps(indicesA, (A + 13 * lda), sizeof(float));
            rowA14 = _mm512_i32gather_ps(indicesA, (A + 14 * lda), sizeof(float));
            rowA15 = _mm512_i32gather_ps(indicesA, (A + 15 * lda), sizeof(float));
        }

        // 8x8 transpose micro kernel

        // Solve the 2x2 diagonal blocks.
        // unpack doesn't work like it does for doubles.
        // Set up for the 2x2 solve.
        __m512 const row0_iter1  = _mm512_unpacklo_ps(rowA0, rowA1);
        __m512 const row1_iter1  = _mm512_unpackhi_ps(rowA0, rowA1);
        __m512 const row2_iter1  = _mm512_unpacklo_ps(rowA2, rowA3);
        __m512 const row3_iter1  = _mm512_unpackhi_ps(rowA2, rowA3);
        __m512 const row4_iter1  = _mm512_unpacklo_ps(rowA4, rowA5);
        __m512 const row5_iter1  = _mm512_unpackhi_ps(rowA4, rowA5);
        __m512 const row6_iter1  = _mm512_unpacklo_ps(rowA6, rowA7);
        __m512 const row7_iter1  = _mm512_unpackhi_ps(rowA6, rowA7);
        __m512 const row8_iter1  = _mm512_unpacklo_ps(rowA8, rowA9);
        __m512 const row9_iter1  = _mm512_unpackhi_ps(rowA8, rowA9);
        __m512 const row10_iter1 = _mm512_unpacklo_ps(rowA10, rowA11);
        __m512 const row11_iter1 = _mm512_unpackhi_ps(rowA10, rowA11);
        __m512 const row12_iter1 = _mm512_unpacklo_ps(rowA12, rowA13);
        __m512 const row13_iter1 = _mm512_unpackhi_ps(rowA12, rowA13);
        __m512 const row14_iter1 = _mm512_unpacklo_ps(rowA14, rowA15);
        __m512 const row15_iter1 = _mm512_unpackhi_ps(rowA14, rowA15);

        // Actually solve the 2x2 blocks.
        __m512 const row0_iter2  = _mm512_shuffle_ps(row0_iter1, row1_iter1, 0x44);
        __m512 const row1_iter2  = _mm512_shuffle_ps(row0_iter1, row1_iter1, 0xee);
        __m512 const row2_iter2  = _mm512_shuffle_ps(row2_iter1, row3_iter1, 0x44);
        __m512 const row3_iter2  = _mm512_shuffle_ps(row2_iter1, row3_iter1, 0xee);
        __m512 const row4_iter2  = _mm512_shuffle_ps(row4_iter1, row5_iter1, 0x44);
        __m512 const row5_iter2  = _mm512_shuffle_ps(row4_iter1, row5_iter1, 0xee);
        __m512 const row6_iter2  = _mm512_shuffle_ps(row6_iter1, row7_iter1, 0x44);
        __m512 const row7_iter2  = _mm512_shuffle_ps(row6_iter1, row7_iter1, 0xee);
        __m512 const row8_iter2  = _mm512_shuffle_ps(row8_iter1, row9_iter1, 0x44);
        __m512 const row9_iter2  = _mm512_shuffle_ps(row8_iter1, row9_iter1, 0xee);
        __m512 const row10_iter2 = _mm512_shuffle_ps(row10_iter1, row11_iter1, 0x44);
        __m512 const row11_iter2 = _mm512_shuffle_ps(row10_iter1, row11_iter1, 0xee);
        __m512 const row12_iter2 = _mm512_shuffle_ps(row12_iter1, row13_iter1, 0x44);
        __m512 const row13_iter2 = _mm512_shuffle_ps(row12_iter1, row13_iter1, 0xee);
        __m512 const row14_iter2 = _mm512_shuffle_ps(row14_iter1, row15_iter1, 0x44);
        __m512 const row15_iter2 = _mm512_shuffle_ps(row14_iter1, row15_iter1, 0xee);

        // Solve the 4x4 diagonal blocks.
        __m512 const row0_iter3  = _mm512_shuffle_ps(row0_iter2, row2_iter2, 0x44);
        __m512 const row1_iter3  = _mm512_shuffle_ps(row1_iter2, row3_iter2, 0x44);
        __m512 const row2_iter3  = _mm512_shuffle_ps(row0_iter2, row2_iter2, 0xee);
        __m512 const row3_iter3  = _mm512_shuffle_ps(row1_iter2, row3_iter2, 0xee);
        __m512 const row4_iter3  = _mm512_shuffle_ps(row4_iter2, row6_iter2, 0x44);
        __m512 const row5_iter3  = _mm512_shuffle_ps(row5_iter2, row7_iter2, 0x44);
        __m512 const row6_iter3  = _mm512_shuffle_ps(row4_iter2, row6_iter2, 0xee);
        __m512 const row7_iter3  = _mm512_shuffle_ps(row5_iter2, row7_iter2, 0xee);
        __m512 const row8_iter3  = _mm512_shuffle_ps(row8_iter2, row10_iter2, 0x44);
        __m512 const row9_iter3  = _mm512_shuffle_ps(row9_iter2, row11_iter2, 0x44);
        __m512 const row10_iter3 = _mm512_shuffle_ps(row8_iter2, row10_iter2, 0xee);
        __m512 const row11_iter3 = _mm512_shuffle_ps(row9_iter2, row11_iter2, 0xee);
        __m512 const row12_iter3 = _mm512_shuffle_ps(row12_iter2, row14_iter2, 0x44);
        __m512 const row13_iter3 = _mm512_shuffle_ps(row13_iter2, row15_iter2, 0x44);
        __m512 const row14_iter3 = _mm512_shuffle_ps(row12_iter2, row14_iter2, 0xee);
        __m512 const row15_iter3 = _mm512_shuffle_ps(row13_iter2, row15_iter2, 0xee);

        // Solve the 8x8 diagonal blocks.
        __m512 const row0_iter4  = _mm512_shuffle_f32x4(row0_iter3, row4_iter3, 0x88);
        __m512 const row1_iter4  = _mm512_shuffle_f32x4(row1_iter3, row5_iter3, 0x88);
        __m512 const row2_iter4  = _mm512_shuffle_f32x4(row2_iter3, row6_iter3, 0x88);
        __m512 const row3_iter4  = _mm512_shuffle_f32x4(row3_iter3, row7_iter3, 0x88);
        __m512 const row4_iter4  = _mm512_shuffle_f32x4(row0_iter3, row4_iter3, 0xdd);
        __m512 const row5_iter4  = _mm512_shuffle_f32x4(row1_iter3, row5_iter3, 0xdd);
        __m512 const row6_iter4  = _mm512_shuffle_f32x4(row2_iter3, row6_iter3, 0xdd);
        __m512 const row7_iter4  = _mm512_shuffle_f32x4(row3_iter3, row7_iter3, 0xdd);
        __m512 const row8_iter4  = _mm512_shuffle_f32x4(row8_iter3, row12_iter3, 0x88);
        __m512 const row9_iter4  = _mm512_shuffle_f32x4(row9_iter3, row13_iter3, 0x88);
        __m512 const row10_iter4 = _mm512_shuffle_f32x4(row10_iter3, row14_iter3, 0x88);
        __m512 const row11_iter4 = _mm512_shuffle_f32x4(row11_iter3, row15_iter3, 0x88);
        __m512 const row12_iter4 = _mm512_shuffle_f32x4(row8_iter3, row12_iter3, 0xdd);
        __m512 const row13_iter4 = _mm512_shuffle_f32x4(row9_iter3, row13_iter3, 0xdd);
        __m512 const row14_iter4 = _mm512_shuffle_f32x4(row10_iter3, row14_iter3, 0xdd);
        __m512 const row15_iter4 = _mm512_shuffle_f32x4(row11_iter3, row15_iter3, 0xdd);

        // Solve the rest.
        rowA0 = _mm512_shuffle_f32x4(row0_iter4, row8_iter4, 0x88);
        rowA1 = _mm512_shuffle_f32x4(row1_iter4, row9_iter4, 0x88);
        rowA2 = _mm512_shuffle_f32x4(row2_iter4, row10_iter4, 0x88);
        rowA3 = _mm512_shuffle_f32x4(row3_iter4, row11_iter4, 0x88);
        rowA4 = _mm512_shuffle_f32x4(row4_iter4, row12_iter4, 0x88);
        rowA5 = _mm512_shuffle_f32x4(row5_iter4, row13_iter4, 0x88);
        rowA6 = _mm512_shuffle_f32x4(row6_iter4, row14_iter4, 0x88);
        rowA7 = _mm512_shuffle_f32x4(row7_iter4, row15_iter4, 0x88);
        rowA8 = _mm512_shuffle_f32x4(row0_iter4, row8_iter4, 0xdd);
        rowA9 = _mm512_shuffle_f32x4(row1_iter4, row9_iter4, 0xdd);
        rowA10 = _mm512_shuffle_f32x4(row2_iter4, row10_iter4, 0xdd);
        rowA11 = _mm512_shuffle_f32x4(row3_iter4, row11_iter4, 0xdd);
        rowA12 = _mm512_shuffle_f32x4(row4_iter4, row12_iter4, 0xdd);
        rowA13 = _mm512_shuffle_f32x4(row5_iter4, row13_iter4, 0xdd);
        rowA14 = _mm512_shuffle_f32x4(row6_iter4, row14_iter4, 0xdd);
        rowA15 = _mm512_shuffle_f32x4(row7_iter4, row15_iter4, 0xdd);

        // Scale A
        rowA0  = _mm512_mul_ps(rowA0, reg_alpha);
        rowA1  = _mm512_mul_ps(rowA1, reg_alpha);
        rowA2  = _mm512_mul_ps(rowA2, reg_alpha);
        rowA3  = _mm512_mul_ps(rowA3, reg_alpha);
        rowA4  = _mm512_mul_ps(rowA4, reg_alpha);
        rowA5  = _mm512_mul_ps(rowA5, reg_alpha);
        rowA6  = _mm512_mul_ps(rowA6, reg_alpha);
        rowA7  = _mm512_mul_ps(rowA7, reg_alpha);
        rowA8  = _mm512_mul_ps(rowA8, reg_alpha);
        rowA9  = _mm512_mul_ps(rowA9, reg_alpha);
        rowA10 = _mm512_mul_ps(rowA10, reg_alpha);
        rowA11 = _mm512_mul_ps(rowA11, reg_alpha);
        rowA12 = _mm512_mul_ps(rowA12, reg_alpha);
        rowA13 = _mm512_mul_ps(rowA13, reg_alpha);
        rowA14 = _mm512_mul_ps(rowA14, reg_alpha);
        rowA15 = _mm512_mul_ps(rowA15, reg_alpha);

        // Load B
        if (!betaIsZero) {
            __m512  rowB0, rowB1, rowB2, rowB3, rowB4, rowB5, rowB6, rowB7, rowB8, rowB9, rowB10, rowB11, rowB12, rowB13, rowB14, rowB15;
            __m512i indicesB;
            if (innerStrideB != 1) {
                indicesB = _mm512_set_epi32(15 * innerStrideB, 14 * innerStrideB, 13 * innerStrideB, 12 * innerStrideB, 11 * innerStrideB,
                                            10 * innerStrideB, 9 * innerStrideB, 8 * innerStrideB, 7 * innerStrideB, 6 * innerStrideB,
                                            5 * innerStrideB, 4 * innerStrideB, 3 * innerStrideB, 2 * innerStrideB, 1 * innerStrideB,
                                            0 * innerStrideB);
                rowB0    = _mm512_i32gather_ps(indicesB, (B + 0 * ldb), sizeof(float));
                rowB1    = _mm512_i32gather_ps(indicesB, (B + 1 * ldb), sizeof(float));
                rowB2    = _mm512_i32gather_ps(indicesB, (B + 2 * ldb), sizeof(float));
                rowB3    = _mm512_i32gather_ps(indicesB, (B + 3 * ldb), sizeof(float));
                rowB4    = _mm512_i32gather_ps(indicesB, (B + 4 * ldb), sizeof(float));
                rowB5    = _mm512_i32gather_ps(indicesB, (B + 5 * ldb), sizeof(float));
                rowB6    = _mm512_i32gather_ps(indicesB, (B + 6 * ldb), sizeof(float));
                rowB7    = _mm512_i32gather_ps(indicesB, (B + 7 * ldb), sizeof(float));
                rowB8    = _mm512_i32gather_ps(indicesB, (B + 8 * ldb), sizeof(float));
                rowB9    = _mm512_i32gather_ps(indicesB, (B + 9 * ldb), sizeof(float));
                rowB10   = _mm512_i32gather_ps(indicesB, (B + 10 * ldb), sizeof(float));
                rowB11   = _mm512_i32gather_ps(indicesB, (B + 11 * ldb), sizeof(float));
                rowB12   = _mm512_i32gather_ps(indicesB, (B + 12 * ldb), sizeof(float));
                rowB13   = _mm512_i32gather_ps(indicesB, (B + 13 * ldb), sizeof(float));
                rowB14   = _mm512_i32gather_ps(indicesB, (B + 14 * ldb), sizeof(float));
                rowB15   = _mm512_i32gather_ps(indicesB, (B + 15 * ldb), sizeof(float));
            } else {
                rowB0  = _mm512_loadu_ps((B + 0 * ldb));
                rowB1  = _mm512_loadu_ps((B + 1 * ldb));
                rowB2  = _mm512_loadu_ps((B + 2 * ldb));
                rowB3  = _mm512_loadu_ps((B + 3 * ldb));
                rowB4  = _mm512_loadu_ps((B + 4 * ldb));
                rowB5  = _mm512_loadu_ps((B + 5 * ldb));
                rowB6  = _mm512_loadu_ps((B + 6 * ldb));
                rowB7  = _mm512_loadu_ps((B + 7 * ldb));
                rowB8  = _mm512_loadu_ps((B + 8 * ldb));
                rowB9  = _mm512_loadu_ps((B + 9 * ldb));
                rowB10 = _mm512_loadu_ps((B + 10 * ldb));
                rowB11 = _mm512_loadu_ps((B + 11 * ldb));
                rowB12 = _mm512_loadu_ps((B + 12 * ldb));
                rowB13 = _mm512_loadu_ps((B + 13 * ldb));
                rowB14 = _mm512_loadu_ps((B + 14 * ldb));
                rowB15 = _mm512_loadu_ps((B + 15 * ldb));
            }

            __m512 const reg_beta = _mm512_set1_ps(beta); // do not alter the content of B

            rowB0  = _mm512_fmadd_ps(reg_beta, rowB0, rowA0);
            rowB1  = _mm512_fmadd_ps(reg_beta, rowB1, rowA1);
            rowB2  = _mm512_fmadd_ps(reg_beta, rowB2, rowA2);
            rowB3  = _mm512_fmadd_ps(reg_beta, rowB3, rowA3);
            rowB4  = _mm512_fmadd_ps(reg_beta, rowB4, rowA4);
            rowB5  = _mm512_fmadd_ps(reg_beta, rowB5, rowA5);
            rowB6  = _mm512_fmadd_ps(reg_beta, rowB6, rowA6);
            rowB7  = _mm512_fmadd_ps(reg_beta, rowB7, rowA7);
            rowB8  = _mm512_fmadd_ps(reg_beta, rowB8, rowA8);
            rowB9  = _mm512_fmadd_ps(reg_beta, rowB9, rowA9);
            rowB10 = _mm512_fmadd_ps(reg_beta, rowB10, rowA10);
            rowB11 = _mm512_fmadd_ps(reg_beta, rowB11, rowA11);
            rowB12 = _mm512_fmadd_ps(reg_beta, rowB12, rowA12);
            rowB13 = _mm512_fmadd_ps(reg_beta, rowB13, rowA13);
            rowB14 = _mm512_fmadd_ps(reg_beta, rowB14, rowA14);
            rowB15 = _mm512_fmadd_ps(reg_beta, rowB15, rowA15);

            // Store B
            if (innerStrideB != 1) {
                _mm512_i32scatter_ps((B + 0 * ldb), indicesB, rowB0, sizeof(float));
                _mm512_i32scatter_ps((B + 1 * ldb), indicesB, rowB1, sizeof(float));
                _mm512_i32scatter_ps((B + 2 * ldb), indicesB, rowB2, sizeof(float));
                _mm512_i32scatter_ps((B + 3 * ldb), indicesB, rowB3, sizeof(float));
                _mm512_i32scatter_ps((B + 4 * ldb), indicesB, rowB4, sizeof(float));
                _mm512_i32scatter_ps((B + 5 * ldb), indicesB, rowB5, sizeof(float));
                _mm512_i32scatter_ps((B + 6 * ldb), indicesB, rowB6, sizeof(float));
                _mm512_i32scatter_ps((B + 7 * ldb), indicesB, rowB7, sizeof(float));
                _mm512_i32scatter_ps((B + 8 * ldb), indicesB, rowB8, sizeof(float));
                _mm512_i32scatter_ps((B + 9 * ldb), indicesB, rowB9, sizeof(float));
                _mm512_i32scatter_ps((B + 10 * ldb), indicesB, rowB10, sizeof(float));
                _mm512_i32scatter_ps((B + 11 * ldb), indicesB, rowB11, sizeof(float));
                _mm512_i32scatter_ps((B + 12 * ldb), indicesB, rowB12, sizeof(float));
                _mm512_i32scatter_ps((B + 13 * ldb), indicesB, rowB13, sizeof(float));
                _mm512_i32scatter_ps((B + 14 * ldb), indicesB, rowB14, sizeof(float));
                _mm512_i32scatter_ps((B + 15 * ldb), indicesB, rowB15, sizeof(float));
            } else {
                _mm512_storeu_ps((B + 0 * ldb), rowB0);
                _mm512_storeu_ps((B + 1 * ldb), rowB1);
                _mm512_storeu_ps((B + 2 * ldb), rowB2);
                _mm512_storeu_ps((B + 3 * ldb), rowB3);
                _mm512_storeu_ps((B + 4 * ldb), rowB4);
                _mm512_storeu_ps((B + 5 * ldb), rowB5);
                _mm512_storeu_ps((B + 6 * ldb), rowB6);
                _mm512_storeu_ps((B + 7 * ldb), rowB7);
                _mm512_storeu_ps((B + 8 * ldb), rowB8);
                _mm512_storeu_ps((B + 9 * ldb), rowB9);
                _mm512_storeu_ps((B + 10 * ldb), rowB10);
                _mm512_storeu_ps((B + 11 * ldb), rowB11);
                _mm512_storeu_ps((B + 12 * ldb), rowB12);
                _mm512_storeu_ps((B + 13 * ldb), rowB13);
                _mm512_storeu_ps((B + 14 * ldb), rowB14);
                _mm512_storeu_ps((B + 15 * ldb), rowB15);
            }
        } else {
            if (innerStrideB != 1) {
                __m512i const indicesB = _mm512_set_epi32(15 * innerStrideB, 14 * innerStrideB, 13 * innerStrideB, 12 * innerStrideB,
                                                          11 * innerStrideB, 10 * innerStrideB, 9 * innerStrideB, 8 * innerStrideB,
                                                          7 * innerStrideB, 6 * innerStrideB, 5 * innerStrideB, 4 * innerStrideB,
                                                          3 * innerStrideB, 2 * innerStrideB, 1 * innerStrideB, 0 * innerStrideB);
                _mm512_i32scatter_ps((B + 0 * ldb), indicesB, rowA0, sizeof(float));
                _mm512_i32scatter_ps((B + 1 * ldb), indicesB, rowA1, sizeof(float));
                _mm512_i32scatter_ps((B + 2 * ldb), indicesB, rowA2, sizeof(float));
                _mm512_i32scatter_ps((B + 3 * ldb), indicesB, rowA3, sizeof(float));
                _mm512_i32scatter_ps((B + 4 * ldb), indicesB, rowA4, sizeof(float));
                _mm512_i32scatter_ps((B + 5 * ldb), indicesB, rowA5, sizeof(float));
                _mm512_i32scatter_ps((B + 6 * ldb), indicesB, rowA6, sizeof(float));
                _mm512_i32scatter_ps((B + 7 * ldb), indicesB, rowA7, sizeof(float));
                _mm512_i32scatter_ps((B + 8 * ldb), indicesB, rowA8, sizeof(float));
                _mm512_i32scatter_ps((B + 9 * ldb), indicesB, rowA9, sizeof(float));
                _mm512_i32scatter_ps((B + 10 * ldb), indicesB, rowA10, sizeof(float));
                _mm512_i32scatter_ps((B + 11 * ldb), indicesB, rowA11, sizeof(float));
                _mm512_i32scatter_ps((B + 12 * ldb), indicesB, rowA12, sizeof(float));
                _mm512_i32scatter_ps((B + 13 * ldb), indicesB, rowA13, sizeof(float));
                _mm512_i32scatter_ps((B + 14 * ldb), indicesB, rowA14, sizeof(float));
                _mm512_i32scatter_ps((B + 15 * ldb), indicesB, rowA15, sizeof(float));
            } else {
                _mm512_storeu_ps((B + 0 * ldb), rowA0);
                _mm512_storeu_ps((B + 1 * ldb), rowA1);
                _mm512_storeu_ps((B + 2 * ldb), rowA2);
                _mm512_storeu_ps((B + 3 * ldb), rowA3);
                _mm512_storeu_ps((B + 4 * ldb), rowA4);
                _mm512_storeu_ps((B + 5 * ldb), rowA5);
                _mm512_storeu_ps((B + 6 * ldb), rowA6);
                _mm512_storeu_ps((B + 7 * ldb), rowA7);
                _mm512_storeu_ps((B + 8 * ldb), rowA8);
                _mm512_storeu_ps((B + 9 * ldb), rowA9);
                _mm512_storeu_ps((B + 10 * ldb), rowA10);
                _mm512_storeu_ps((B + 11 * ldb), rowA11);
                _mm512_storeu_ps((B + 12 * ldb), rowA12);
                _mm512_storeu_ps((B + 13 * ldb), rowA13);
                _mm512_storeu_ps((B + 14 * ldb), rowA14);
                _mm512_storeu_ps((B + 15 * ldb), rowA15);
            }
        }
    }
};

template <>
void streamingStore<float>(float *out, float const *in) {
    _mm512_stream_ps(out, _mm512_loadu_ps(in));
}
template <>
void streamingStore<double>(double *out, double const *in) {
    _mm512_stream_pd(out, _mm512_loadu_pd(in));
}
