//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <immintrin.h>

#ifdef EINSUMS_DEBUG
#ifdef __SSE2__
#warning "DEBUG INFO: Compiling HPTT with SSE2 as the highest level of vectorization."
#else
#warning "DEBUG INFO: Compiling HPTT with SSE1 as the highest level of vectorization."
#endif
#endif

#ifdef __SSE2__
template <bool betaIsZero, bool conjA>
struct micro_kernel<double, betaIsZero, conjA> {
    static void execute(double const *A, size_t const lda, size_t const innerStrideA, double *B, size_t const ldb,
                        size_t const innerStrideB, double const alpha, double const beta) {
        __m128d alpha_reg = _mm_set1_pd(alpha);

        __m128d rowA0, rowA1;

        if (innerStrideA != 1) {
            rowA0 = _mm_loadl_pd(rowA0, (A + 0 * lda));
            rowA0 = _mm_loadh_pd(rowA0, (A + 0 * lda + innerStrideA));
            rowA1 = _mm_loadl_pd(rowA1, (A + 1 * lda));
            rowA1 = _mm_loadh_pd(rowA1, (A + 1 * lda + innerStrideA));
        } else {
            rowA0 = _mm_loadu_pd(A);
            rowA1 = _mm_loadu_pd(A + lda);
        }

        // Transpose.
        __m128d const row0 = _mm_unpacklo_pd(rowA0, rowA1);
        __m128d const row1 = _mm_unpackhi_pd(rowA0, rowA1);

        // Scale.
        rowA0 = _mm_mul_pd(row0, alpha_reg);
        rowA1 = _mm_mul_pd(row1, alpha_reg);

        // Store into B.
        if constexpr (!betaIsZero) {
            __m128d beta_reg = _mm_set1_pd(beta);

            __m128d rowB0, rowB1;

            // Load B.
            if (innerStrideB != 1) {
                rowB0 = _mm_loadl_pd(rowB0, (B + 0 * ldb));
                rowB0 = _mm_loadh_pd(rowB0, (B + 0 * ldb + innerStrideB));
                rowB1 = _mm_loadl_pd(rowB1, (B + 1 * ldb));
                rowB1 = _mm_loadh_pd(rowB1, (B + 1 * ldb + innerStrideB));
            } else {
                rowB0 = _mm_loadu_pd(B);
                rowB1 = _mm_loadu_pd(B + ldb);
            }

            // No FMA instructions in SSE.
            rowB0 = _mm_add_pd(rowA0, _mm_mul_pd(rowB0, beta_reg));
            rowB1 = _mm_add_pd(rowA1, _mm_mul_pd(rowB1, beta_reg));

            // Store.
            if (innerStrideB != 1) {
                _mm_storel_pd((B + 0 * ldb), rowB0);
                _mm_storeh_pd((B + 0 * ldb + innerStrideB), rowB0);
                _mm_storel_pd((B + 1 * ldb), rowB1);
                _mm_storeh_pd((B + 1 * ldb + innerStrideB), rowB1);
            } else {
                _mm_storeu_pd(B, rowB0);
                _mm_storeu_pd(B + ldb, rowB1);
            }
        } else {
            if (innerStrideB != 1) {
                _mm_storel_pd((B + 0 * ldb), rowA0);
                _mm_storeh_pd((B + 0 * ldb + innerStrideB), rowA0);
                _mm_storel_pd((B + 1 * ldb), rowA1);
                _mm_storeh_pd((B + 1 * ldb + innerStrideB), rowA1);
            } else {
                _mm_storeu_pd(B, rowA0);
                _mm_storeu_pd(B + ldb, rowA1);
            }
        }
    }
};
#endif

template <bool betaIsZero, bool conjA>
struct micro_kernel<float, betaIsZero, conjA> {
    static void execute(float const *A, size_t const lda, size_t const innerStrideA, float *B, size_t const ldb, size_t const innerStrideB,
                        float const alpha, float const beta) {
        __m128 const alpha_reg = _mm_set1_ps(alpha);

        __m128 rowA0, rowA1, rowA2, rowA3;

        if (innerStrideA != 1) {
            rowA0 = _mm_set_ps(A[0 * lda + 3 * innerStrideA], A[0 * lda + 2 * innerStrideA], A[0 * lda + 1 * innerStrideA],
                               A[0 * lda + 0 * innerStrideA]);
            rowA1 = _mm_set_ps(A[1 * lda + 3 * innerStrideA], A[1 * lda + 2 * innerStrideA], A[1 * lda + 1 * innerStrideA],
                               A[1 * lda + 0 * innerStrideA]);
            rowA2 = _mm_set_ps(A[2 * lda + 3 * innerStrideA], A[2 * lda + 2 * innerStrideA], A[2 * lda + 1 * innerStrideA],
                               A[2 * lda + 0 * innerStrideA]);
            rowA3 = _mm_set_ps(A[3 * lda + 3 * innerStrideA], A[3 * lda + 2 * innerStrideA], A[3 * lda + 1 * innerStrideA],
                               A[3 * lda + 0 * innerStrideA]);
        } else {
            rowA0 = _mm_loadu_ps(A + 0 * lda);
            rowA1 = _mm_loadu_ps(A + 1 * lda);
            rowA2 = _mm_loadu_ps(A + 2 * lda);
            rowA3 = _mm_loadu_ps(A + 3 * lda);
        }

        // Transpose.
        _MM_TRANSPOSE4_PS(rowA0, rowA1, rowA2, rowA3);

        // Scale A.
        rowA0 = _mm_mul_ps(rowA0, alpha_reg);
        rowA1 = _mm_mul_ps(rowA1, alpha_reg);
        rowA2 = _mm_mul_ps(rowA2, alpha_reg);
        rowA3 = _mm_mul_ps(rowA3, alpha_reg);

        // Store into B.
        if constexpr (!betaIsZero) {
            __m128 const beta_reg = _mm_set1_ps(beta);

            __m128 rowB0, rowB1, rowB2, rowB3;

            // Load B.
            if (innerStrideB != 1) {
                rowB0 = _mm_set_ps(B[0 * ldb + 3 * innerStrideB], B[0 * ldb + 2 * innerStrideB], B[0 * ldb + 1 * innerStrideB],
                                   B[0 * ldb + 0 * innerStrideB]);
                rowB1 = _mm_set_ps(B[1 * ldb + 3 * innerStrideB], B[1 * ldb + 2 * innerStrideB], B[1 * ldb + 1 * innerStrideB],
                                   B[1 * ldb + 0 * innerStrideB]);
                rowB2 = _mm_set_ps(B[2 * ldb + 3 * innerStrideB], B[2 * ldb + 2 * innerStrideB], B[2 * ldb + 1 * innerStrideB],
                                   B[2 * ldb + 0 * innerStrideB]);
                rowB3 = _mm_set_ps(B[3 * ldb + 3 * innerStrideB], B[3 * ldb + 2 * innerStrideB], B[3 * ldb + 1 * innerStrideB],
                                   B[3 * ldb + 0 * innerStrideB]);
            } else {
                rowB0 = _mm_loadu_ps(B + 0 * ldb);
                rowB1 = _mm_loadu_ps(B + 1 * ldb);
                rowB2 = _mm_loadu_ps(B + 2 * ldb);
                rowB3 = _mm_loadu_ps(B + 3 * ldb);
            }

            // Scale B
            rowB0 = _mm_add_ps(rowA0, _mm_mul_ps(rowB0, beta_reg));
            rowB1 = _mm_add_ps(rowA1, _mm_mul_ps(rowB1, beta_reg));
            rowB2 = _mm_add_ps(rowA2, _mm_mul_ps(rowB2, beta_reg));
            rowB3 = _mm_add_ps(rowA3, _mm_mul_ps(rowB3, beta_reg));

            // Store into B.
            if (innerStrideB != 1) {
                float hold[4];

                _mm_storeu_ps(hold, rowB0);

                for (int i = 0; i < 4; i++) {
                    B[0 * ldb + i * innerStrideB] = hold[i];
                }

                _mm_storeu_ps(hold, rowB1);

                for (int i = 0; i < 4; i++) {
                    B[1 * ldb + i * innerStrideB] = hold[i];
                }

                _mm_storeu_ps(hold, rowB2);

                for (int i = 0; i < 4; i++) {
                    B[2 * ldb + i * innerStrideB] = hold[i];
                }

                _mm_storeu_ps(hold, rowB3);

                for (int i = 0; i < 4; i++) {
                    B[3 * ldb + i * innerStrideB] = hold[i];
                }
            } else {
                _mm_storeu_ps(B + 0 * ldb, rowB0);
                _mm_storeu_ps(B + 1 * ldb, rowB1);
                _mm_storeu_ps(B + 2 * ldb, rowB2);
                _mm_storeu_ps(B + 3 * ldb, rowB3);
            }
        } else {
            if (innerStrideB != 1) {
                float hold[4];

                _mm_storeu_ps(hold, rowA0);

                for (int i = 0; i < 4; i++) {
                    B[0 * ldb + i * innerStrideB] = hold[i];
                }

                _mm_storeu_ps(hold, rowA1);

                for (int i = 0; i < 4; i++) {
                    B[1 * ldb + i * innerStrideB] = hold[i];
                }

                _mm_storeu_ps(hold, rowA2);

                for (int i = 0; i < 4; i++) {
                    B[2 * ldb + i * innerStrideB] = hold[i];
                }

                _mm_storeu_ps(hold, rowA3);

                for (int i = 0; i < 4; i++) {
                    B[3 * ldb + i * innerStrideB] = hold[i];
                }
            } else {
                _mm_storeu_ps(B + 0 * ldb, rowA0);
                _mm_storeu_ps(B + 1 * ldb, rowA1);
                _mm_storeu_ps(B + 2 * ldb, rowA2);
                _mm_storeu_ps(B + 3 * ldb, rowA3);
            }
        }
    }
};

template <>
void streamingStore<float>(float *out, float const *in) {
    _mm_stream_ps(out, _mm_loadu_ps(in));
}
template <>
void streamingStore<double>(double *out, double const *in) {
    _mm_stream_pd(out, _mm_loadu_pd(in));
}