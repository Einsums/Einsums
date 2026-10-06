//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------
#include <arm_neon.h>

#ifdef EINSUMS_DEBUG
#warning "DEBUG INFO: Compiling HPTT with ARM Neon as the highest level of vectorization."
#endif

template <bool betaIsZero, bool conjA>
struct micro_kernel<float, betaIsZero, conjA> {
    static void execute(float const *A, size_t const lda, size_t const innerStrideA, float *B, size_t const ldb, size_t const innerStrideB,
                        float const alpha, float const beta) {
        float32x4_t const reg_alpha = vdupq_n_f32(alpha);

        // Load A
        float32x4_t rowA0, rowA1, rowA2, rowA3;
        if (innerStrideA == 1) {
            rowA0 = vld1q_f32((A + 0 * lda));
            rowA1 = vld1q_f32((A + 1 * lda));
            rowA2 = vld1q_f32((A + 2 * lda));
            rowA3 = vld1q_f32((A + 3 * lda));
        } else if (innerStrideA == 2) {
            rowA0 = vld2q_f32((A + 0 * lda)).val[0];
            rowA1 = vld2q_f32((A + 1 * lda)).val[0];
            rowA2 = vld2q_f32((A + 2 * lda)).val[0];
            rowA3 = vld2q_f32((A + 3 * lda)).val[0];
        } else if (innerStrideA == 3) {
            rowA0 = vld3q_f32((A + 0 * lda)).val[0];
            rowA1 = vld3q_f32((A + 1 * lda)).val[0];
            rowA2 = vld3q_f32((A + 2 * lda)).val[0];
            rowA3 = vld3q_f32((A + 3 * lda)).val[0];
        } else if (innerStrideA == 4) {
            rowA0 = vld4q_f32((A + 0 * lda)).val[0];
            rowA1 = vld4q_f32((A + 1 * lda)).val[0];
            rowA2 = vld4q_f32((A + 2 * lda)).val[0];
            rowA3 = vld4q_f32((A + 3 * lda)).val[0];
        } else {
            rowA0 = vdupq_n_f32(0);
            rowA1 = vdupq_n_f32(0);
            rowA2 = vdupq_n_f32(0);
            rowA3 = vdupq_n_f32(0);

            rowA0 = vld1q_lane_f32(A + 0 * lda + 0 * innerStrideA, rowA0, 0);
            rowA0 = vld1q_lane_f32(A + 0 * lda + 1 * innerStrideA, rowA0, 1);
            rowA0 = vld1q_lane_f32(A + 0 * lda + 2 * innerStrideA, rowA0, 2);
            rowA0 = vld1q_lane_f32(A + 0 * lda + 3 * innerStrideA, rowA0, 3);

            rowA1 = vld1q_lane_f32(A + 1 * lda + 0 * innerStrideA, rowA1, 0);
            rowA1 = vld1q_lane_f32(A + 1 * lda + 1 * innerStrideA, rowA1, 1);
            rowA1 = vld1q_lane_f32(A + 1 * lda + 2 * innerStrideA, rowA1, 2);
            rowA1 = vld1q_lane_f32(A + 1 * lda + 3 * innerStrideA, rowA1, 3);

            rowA2 = vld1q_lane_f32(A + 2 * lda + 0 * innerStrideA, rowA2, 0);
            rowA2 = vld1q_lane_f32(A + 2 * lda + 1 * innerStrideA, rowA2, 1);
            rowA2 = vld1q_lane_f32(A + 2 * lda + 2 * innerStrideA, rowA2, 2);
            rowA2 = vld1q_lane_f32(A + 2 * lda + 3 * innerStrideA, rowA2, 3);

            rowA3 = vld1q_lane_f32(A + 3 * lda + 0 * innerStrideA, rowA3, 0);
            rowA3 = vld1q_lane_f32(A + 3 * lda + 1 * innerStrideA, rowA3, 1);
            rowA3 = vld1q_lane_f32(A + 3 * lda + 2 * innerStrideA, rowA3, 2);
            rowA3 = vld1q_lane_f32(A + 3 * lda + 3 * innerStrideA, rowA3, 3);
        }

        // 4x4 transpose micro kernel
        float32x4x2_t const t0 = vuzpq_f32(rowA0, rowA2);
        float32x4x2_t const t1 = vuzpq_f32(rowA1, rowA3);
        float32x4x2_t const t2 = vtrnq_f32(t0.val[0], t1.val[0]);
        float32x4x2_t const t3 = vtrnq_f32(t0.val[1], t1.val[1]);

        // Scale A
        rowA0 = vmulq_f32(t2.val[0], reg_alpha);
        rowA1 = vmulq_f32(t3.val[0], reg_alpha);
        rowA2 = vmulq_f32(t2.val[1], reg_alpha);
        rowA3 = vmulq_f32(t3.val[1], reg_alpha);

        // Load B
        if constexpr (!betaIsZero) {
            float32x4_t rowB0, rowB1, rowB2, rowB3;
            float32x4_t const reg_beta  = vdupq_n_f32(beta);
            if (innerStrideB == 1) {
                rowB0 = vld1q_f32((B + 0 * ldb));
                rowB1 = vld1q_f32((B + 1 * ldb));
                rowB2 = vld1q_f32((B + 2 * ldb));
                rowB3 = vld1q_f32((B + 3 * ldb));
            } else if (innerStrideB == 2) {
                rowB0 = vld2q_f32((B + 0 * ldb)).val[0];
                rowB1 = vld2q_f32((B + 1 * ldb)).val[0];
                rowB2 = vld2q_f32((B + 2 * ldb)).val[0];
                rowB3 = vld2q_f32((B + 3 * ldb)).val[0];
            } else if (innerStrideB == 3) {
                rowB0 = vld3q_f32((B + 0 * ldb)).val[0];
                rowB1 = vld3q_f32((B + 1 * ldb)).val[0];
                rowB2 = vld3q_f32((B + 2 * ldb)).val[0];
                rowB3 = vld3q_f32((B + 3 * ldb)).val[0];
            } else if (innerStrideB == 4) {
                rowB0 = vld4q_f32((B + 0 * ldb)).val[0];
                rowB1 = vld4q_f32((B + 1 * ldb)).val[0];
                rowB2 = vld4q_f32((B + 2 * ldb)).val[0];
                rowB3 = vld4q_f32((B + 3 * ldb)).val[0];
            } else {
                rowB0 = vdupq_n_f32(0);
                rowB1 = vdupq_n_f32(0);
                rowB2 = vdupq_n_f32(0);
                rowB3 = vdupq_n_f32(0);

                rowB0 = vld1q_lane_f32(B + 0 * innerStrideB, rowB0, 0);
                rowB0 = vld1q_lane_f32(B + 1 * innerStrideB, rowB0, 1);
                rowB0 = vld1q_lane_f32(B + 2 * innerStrideB, rowB0, 2);
                rowB0 = vld1q_lane_f32(B + 3 * innerStrideB, rowB0, 3);

                rowB1 = vld1q_lane_f32(B + 1 * ldb + 0 * innerStrideB, rowB1, 0);
                rowB1 = vld1q_lane_f32(B + 1 * ldb + 1 * innerStrideB, rowB1, 1);
                rowB1 = vld1q_lane_f32(B + 1 * ldb + 2 * innerStrideB, rowB1, 2);
                rowB1 = vld1q_lane_f32(B + 1 * ldb + 3 * innerStrideB, rowB1, 3);

                rowB2 = vld1q_lane_f32(B + 2 * ldb + 0 * innerStrideB, rowB2, 0);
                rowB2 = vld1q_lane_f32(B + 2 * ldb + 1 * innerStrideB, rowB2, 1);
                rowB2 = vld1q_lane_f32(B + 2 * ldb + 2 * innerStrideB, rowB2, 2);
                rowB2 = vld1q_lane_f32(B + 2 * ldb + 3 * innerStrideB, rowB2, 3);

                rowB3 = vld1q_lane_f32(B + 3 * ldb + 0 * innerStrideB, rowB3, 0);
                rowB3 = vld1q_lane_f32(B + 3 * ldb + 1 * innerStrideB, rowB3, 1);
                rowB3 = vld1q_lane_f32(B + 3 * ldb + 2 * innerStrideB, rowB3, 2);
                rowB3 = vld1q_lane_f32(B + 3 * ldb + 3 * innerStrideB, rowB3, 3);
            }

            rowB0 = vfmaq_f32(rowA0, rowB0, reg_beta);
            rowB1 = vfmaq_f32(rowA1, rowB1, reg_beta);
            rowB2 = vfmaq_f32(rowA2, rowB2, reg_beta);
            rowB3 = vfmaq_f32(rowA3, rowB3, reg_beta);
            
            // Store B
            if (innerStrideB == 1) {
                vst1q_f32((B + 0 * ldb), rowB0);
                vst1q_f32((B + 1 * ldb), rowB1);
                vst1q_f32((B + 2 * ldb), rowB2);
                vst1q_f32((B + 3 * ldb), rowB3);
            } else {
                float tmp[4];
                vst1q_f32(tmp, rowB0);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB] = tmp[i];
                }
                vst1q_f32(tmp, rowB1);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB + 1 * ldb] = tmp[i];
                }
                vst1q_f32(tmp, rowB2);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB + 2 * ldb] = tmp[i];
                }
                vst1q_f32(tmp, rowB3);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB + 3 * ldb] = tmp[i];
                }
            }
        } else {
            // Store B
            if (innerStrideB == 1) {
                vst1q_f32((B + 0 * ldb), rowA0);
                vst1q_f32((B + 1 * ldb), rowA1);
                vst1q_f32((B + 2 * ldb), rowA2);
                vst1q_f32((B + 3 * ldb), rowA3);
            } else {
                float tmp[4];
                vst1q_f32(tmp, rowA0);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB] = tmp[i];
                }
                vst1q_f32(tmp, rowA1);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB + 1 * ldb] = tmp[i];
                }
                vst1q_f32(tmp, rowA2);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB + 2 * ldb] = tmp[i];
                }
                vst1q_f32(tmp, rowA3);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    B[i * innerStrideB + 3 * ldb] = tmp[i];
                }
            }
        }
    }
};