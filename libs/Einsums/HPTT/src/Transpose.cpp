//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/*
  Copyright 2018 Paul Springer

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * \file
 * Compute the tensor transposition
 */

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <float.h>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

#include <Einsums/HPTT/ComputeNode.hpp>
#include <Einsums/HPTT/Files.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Macros.hpp>
#include <Einsums/HPTT/Plan.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/HPTT/Utils.hpp>

#ifdef __x86_64__
#    include <immintrin.h>
#endif

namespace hptt {

template <typename floatType, bool betaIsZero, bool conjA>
struct micro_kernel {
    static void execute(floatType const *A, size_t const lda, size_t const innerStrideA, floatType *B, size_t const ldb,
                        size_t const innerStrideB, floatType const alpha, floatType const beta) {
        constexpr size_t n = (REGISTER_BITS / 8) / sizeof(floatType);

        if constexpr (betaIsZero) {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    if constexpr (conjA)
                        B[(i * innerStrideB) + (j * ldb)] = alpha * conj(A[(j * innerStrideA) + (lda * i)]);
                    else
                        B[(i * innerStrideB) + (j * ldb)] = alpha * A[(j * innerStrideA) + (lda * i)];
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < n; ++i) {
                    if constexpr (conjA) {
                        B[(i * innerStrideB) + (j * ldb)] =
                            alpha * conj(A[(j * innerStrideA) + (lda * i)]) + beta * B[(i * innerStrideB) + (j * ldb)];
                    } else {
                        B[(i * innerStrideB) + (j * ldb)] =
                            alpha * A[(j * innerStrideA) + (lda * i)] + beta * B[(i * innerStrideB) + (j * ldb)];
                    }
                }
            }
        }
    }
};

template <typename floatType>
static void streamingStore(floatType *out, floatType const *in) {
    constexpr int n = REGISTER_BITS / 8 / sizeof(floatType);
    for (int i = 0; i < n; ++i)
        out[i] = in[i];
}

#ifdef __SSE__
template <typename floatType>
static INLINE void prefetch(floatType const *A, size_t const lda) {
    constexpr size_t blocking_micro_ = REGISTER_BITS / 8 / sizeof(floatType);
    for (size_t i = 0; i < blocking_micro_; ++i)
        _mm_prefetch((char *)(A + i * lda), _MM_HINT_T2);
}
#else
template <typename floatType>
static INLINE void prefetch(floatType const *A, size_t const lda) {
}
#endif

#if defined(__x86_64__) && defined(__AVX__) && defined(__AVX2__) && defined(__AVX512F__) && defined(__AVX512VL__)
#    include "microkernels/avx512.hpp"
#elif defined(__x86_64) && defined(__AVX__)
#    include "microkernels/avx.hpp"
#elif defined(__x86_64__) && defined(__SSE__)
#    include "microkernels/sse.hpp"
#elif defined(__aarch64) && defined(__ARM_NEON)
#    include "microkernels/arm_neon.hpp"
#endif

template <bool betaIsZero, typename floatType, bool conjA>
static INLINE void macro_kernel_scalar(floatType const *A, size_t const lda, int blockingA, size_t innerStrideA, floatType *B,
                                       size_t const ldb, int blockingB, size_t innerStrideB, floatType const alpha, floatType const beta) {
#ifdef DEBUG
    assert(blockingA > 0 && blockingB > 0);
#endif

    if constexpr (betaIsZero) {
        for (int j = 0; j < blockingA; ++j) {
            for (int i = 0; i < blockingB; ++i) {
                if (conjA)
                    B[(i * innerStrideB) + (j * ldb)] = alpha * conj(A[(i * lda) + (j * innerStrideA)]);
                else
                    B[(i * innerStrideB) + (j * ldb)] = alpha * A[(i * lda) + (j * innerStrideA)];
            }
        }
    } else {
        for (int j = 0; j < blockingA; ++j) {
            for (int i = 0; i < blockingB; ++i) {
                if (conjA)
                    B[(i * innerStrideB) + (j * ldb)] =
                        alpha * conj(A[(i * lda) + (j * innerStrideA)]) + beta * B[(i * innerStrideB) + (j * ldb)];
                else
                    B[(i * innerStrideB) + (j * ldb)] =
                        alpha * A[(i * lda) + (j * innerStrideA)] + beta * B[(i * innerStrideB) + (j * ldb)];
            }
        }
    }
}

template <int blockingA, int blockingB, bool betaIsZero, typename floatType, bool useStreamingStores_, bool conjA>
static INLINE void macro_kernel(floatType const *A, floatType const *Anext, size_t const lda, size_t innerStrideA, floatType *B,
                                floatType const *Bnext, size_t const ldb, size_t innerStrideB, floatType const alpha,
                                floatType const beta) {
    constexpr int blocking_micro_ = REGISTER_BITS / 8 / sizeof(floatType);
    constexpr int blocking_       = blocking_micro_ * 4;

    bool const useStreamingStores = useStreamingStores_ && betaIsZero && (blockingB * sizeof(floatType)) % 64 == 0 &&
                                    ((uint64_t)B) % 32 == 0 && (ldb * sizeof(floatType)) % 32 == 0;

    floatType *Btmp    = B;
    size_t     ldb_tmp = ldb;
    floatType  buffer[blockingA * blockingB]; // __attribute__((aligned(64)));
    if ((useStreamingStores_ && useStreamingStores && innerStrideB == 1)) {
        Btmp    = buffer;
        ldb_tmp = blockingB;
    }

    if constexpr (blockingA == blocking_ && blockingB == blocking_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (0 * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB,
                                                            alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (0 * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (2 * blocking_micro_ * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (2 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (3 * blocking_micro_ * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (3 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                            alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (2 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (0 * lda + 2 * blocking_micro_), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (blocking_micro_ * lda + 2 * blocking_micro_), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (2 * blocking_micro_ * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (2 * blocking_micro_ * lda + 2 * blocking_micro_), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (3 * blocking_micro_ * lda + 2 * blocking_micro_), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (3 * blocking_micro_ * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else if constexpr (blockingA == 2 * blocking_micro_ && blockingB == blocking_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (0 * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB,
                                                            alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (0 * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (2 * blocking_micro_ * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (2 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (3 * blocking_micro_ * lda + (innerStrideA * 0)), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (3 * blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                            alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 2 * blocking_micro_), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (2 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 2 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (3 * blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 3 * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else if constexpr (blockingA == blocking_ && blockingB == 2 * blocking_micro_) {
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (0 * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (0 * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (blocking_micro_ * lda + 0), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (blocking_micro_ * lda + (innerStrideA * 0)), lda, innerStrideA,
                                                            Btmp + (0 * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB,
                                                            alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp, innerStrideB,
                                                            alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * blocking_micro_)), lda, innerStrideA,
            Btmp + (blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (2 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (0 * lda + 2 * blocking_micro_), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * 0)), ldb_tmp,
                                                            innerStrideB, alpha, beta);
        if (innerStrideA == 1)
            prefetch<floatType>(Anext + (blocking_micro_ * lda + 2 * blocking_micro_), lda);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 2 * blocking_micro_)), lda, innerStrideA,
            Btmp + (2 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
        if (!(useStreamingStores_ && useStreamingStores) && innerStrideB == 1)
            prefetch<floatType>(Bnext + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp);
        micro_kernel<floatType, betaIsZero, conjA>::execute(A + (0 * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
                                                            Btmp + (3 * blocking_micro_ * ldb_tmp + 0), ldb_tmp, innerStrideB, alpha, beta);
        micro_kernel<floatType, betaIsZero, conjA>::execute(
            A + (blocking_micro_ * lda + (innerStrideA * 3 * blocking_micro_)), lda, innerStrideA,
            Btmp + (3 * blocking_micro_ * ldb_tmp + (innerStrideB * blocking_micro_)), ldb_tmp, innerStrideB, alpha, beta);
    } else {
        // invoke micro-transpose
        if (blockingA > 0 && blockingB > 0)
            micro_kernel<floatType, betaIsZero, conjA>::execute(A, lda, innerStrideA, Btmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 0 && blockingB > blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + blocking_micro_ * lda, lda, innerStrideA, Btmp + (innerStrideB * blocking_micro_), ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 0 && blockingB > 2 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(A + 2 * blocking_micro_ * lda, lda, innerStrideA,
                                                                Btmp + (innerStrideB * 2 * blocking_micro_), ldb_tmp, innerStrideB, alpha,
                                                                beta);

        // invoke micro-transpose
        if (blockingA > 0 && blockingB > 3 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(A + 3 * blocking_micro_ * lda, lda, innerStrideA,
                                                                Btmp + (innerStrideB * 3 * blocking_micro_), ldb_tmp, innerStrideB, alpha,
                                                                beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > 0)
            micro_kernel<floatType, betaIsZero, conjA>::execute(A + (innerStrideA * blocking_micro_), lda, innerStrideA,
                                                                Btmp + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * blocking_micro_) + blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * blocking_micro_) + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > 2 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * blocking_micro_) + 2 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 2 * blocking_micro_) + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > blocking_micro_ && blockingB > 3 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * blocking_micro_) + 3 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 3 * blocking_micro_) + blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > 0)
            micro_kernel<floatType, betaIsZero, conjA>::execute(A + (innerStrideA * 2 * blocking_micro_), lda, innerStrideA,
                                                                Btmp + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 2 * blocking_micro_) + blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * blocking_micro_) + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > 2 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 2 * blocking_micro_) + 2 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 2 * blocking_micro_) + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 2 * blocking_micro_ && blockingB > 3 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 2 * blocking_micro_) + 3 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 3 * blocking_micro_) + 2 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > 0)
            micro_kernel<floatType, betaIsZero, conjA>::execute(A + (innerStrideA * 3 * blocking_micro_), lda, innerStrideA,
                                                                Btmp + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 3 * blocking_micro_) + blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * blocking_micro_) + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > 2 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 3 * blocking_micro_) + 2 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 2 * blocking_micro_) + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);

        // invoke micro-transpose
        if (blockingA > 3 * blocking_micro_ && blockingB > 3 * blocking_micro_)
            micro_kernel<floatType, betaIsZero, conjA>::execute(
                A + (innerStrideA * 3 * blocking_micro_) + 3 * blocking_micro_ * lda, lda, innerStrideA,
                Btmp + (innerStrideB * 3 * blocking_micro_) + 3 * blocking_micro_ * ldb_tmp, ldb_tmp, innerStrideB, alpha, beta);
    }

    // write buffer to main-memory via non-temporal stores
    if ((useStreamingStores_ && useStreamingStores && innerStrideB == 1)) {
        for (int i = 0; i < blockingA; i++) {
            for (int j = 0; j < blockingB; j += blocking_micro_)
                streamingStore<floatType>(B + i * ldb + j, buffer + i * ldb_tmp + j);
        }
    }
}

template <bool betaIsZero, typename floatType, bool conjA>
void transpose_int_scalar(floatType const *A, size_t sizeStride1A, size_t innerStrideA, floatType *B, size_t sizeStride1B,
                          size_t innerStrideB, floatType const alpha, floatType const beta, ComputeNode const *plan) {
    ptrdiff_t const end       = plan->end;
    size_t const    lda       = plan->lda;
    size_t const    ldb       = plan->ldb;
    ptrdiff_t const offDiffAB = plan->offDiffAB;
    if (plan->next->next != nullptr) {
        // recurse
        ptrdiff_t i = plan->start;
        if (plan->indexA)
            transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], end - plan->start, innerStrideA, &B[i * ldb],
                                                               sizeStride1B, innerStrideB, alpha, beta, plan->next);
        else if (plan->indexB)
            transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], sizeStride1A, innerStrideA, &B[i * ldb],
                                                               end - plan->start, innerStrideB, alpha, beta, plan->next);
        else
            for (; i < end; i++)
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], sizeStride1A, innerStrideA, &B[i * ldb],
                                                                   sizeStride1B, innerStrideB, alpha, beta, plan->next);
    } else {
        // macro-kernel
        size_t const    lda_macro       = plan->next->lda;
        size_t const    ldb_macro       = plan->next->ldb;
        ptrdiff_t       i               = plan->start;
        ptrdiff_t const scalarRemainder = plan->end - plan->start;
        if (scalarRemainder > 0) {
            if (lda == 1)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, scalarRemainder, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, sizeStride1B, innerStrideB, alpha, beta);
            else if (ldb == 1)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, sizeStride1A, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, scalarRemainder, innerStrideB, alpha, beta);
            else
                for (; i < end; i++)
                    macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, sizeStride1A, innerStrideA,
                                                                      &B[i * ldb], ldb_macro, sizeStride1B, innerStrideB, alpha, beta);
        }
    }
}
template <int blockingA, int blockingB, bool betaIsZero, typename floatType, bool useStreamingStores, bool conjA>
void transpose_int(floatType const *A, floatType const *Anext, size_t innerStrideA, floatType *B, floatType const *Bnext,
                   size_t innerStrideB, floatType const alpha, floatType const beta, ComputeNode const *plan) {
    ptrdiff_t const end       = plan->end - (plan->inc - 1);
    ptrdiff_t const inc       = plan->inc;
    size_t const    lda       = plan->lda;
    size_t const    ldb       = plan->ldb;
    int32_t const   offDiffAB = plan->offDiffAB;

    constexpr int blocking_micro_ = REGISTER_BITS / 8 / sizeof(floatType);
    constexpr int blocking_       = blocking_micro_ * 4;

    if (plan->next->next != nullptr) {
        // recurse
        ptrdiff_t i;
        for (i = plan->start; i < end; i += inc) {
            if (i + inc < end)
                transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], &A[(i + 1 + offDiffAB) * lda], innerStrideA, &B[i * ldb], &B[(i + 1) * ldb], innerStrideB,
                    alpha, beta, plan->next);
            else if (i == plan->start || i + inc >= end)
                transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], &A[(i + offDiffAB) * lda], innerStrideA, &B[i * ldb], &B[i * ldb], innerStrideB, alpha, beta,
                    plan->next);
            else
                transpose_int<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next);
        }
        // remainder
        if (blocking_ / 2 >= blocking_micro_ && (i + blocking_ / 2) <= plan->end) {
            if (plan->indexA)
                transpose_int<blocking_ / 2, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next);
            else if (plan->indexB)
                transpose_int<blockingA, blocking_ / 2, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next);
            i += blocking_ / 2;
        }
        if (blocking_ / 4 >= blocking_micro_ && (i + blocking_ / 4) <= plan->end) {
            if (plan->indexA)
                transpose_int<blocking_ / 4, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next);
            else if (plan->indexB)
                transpose_int<blockingA, blocking_ / 4, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, innerStrideA, &B[i * ldb], Bnext, innerStrideB, alpha, beta, plan->next);
            i += blocking_ / 4;
        }
        ptrdiff_t const scalarRemainder = plan->end - i;
        if (scalarRemainder > 0) {
            if (plan->indexA)
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], scalarRemainder, innerStrideA, &B[i * ldb],
                                                                   blockingB, innerStrideB, alpha, beta, plan->next);
            else if (plan->indexB)
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], blockingA, innerStrideA, &B[i * ldb],
                                                                   scalarRemainder, innerStrideB, alpha, beta, plan->next);
            else
                transpose_int_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], blockingA, innerStrideA, &B[i * ldb],
                                                                   blockingB, innerStrideB, alpha, beta, plan->next);
        }
    } else {
        size_t const lda_macro = plan->next->lda;
        size_t const ldb_macro = plan->next->ldb;
        // invoke macro-kernel

        ptrdiff_t i;
        for (i = plan->start; i < end; i += inc)
            if (i + inc < end)
                macro_kernel<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], &A[(i + 1) * lda], lda_macro, innerStrideA, &B[i * ldb], &B[(i + 1) * ldb], ldb_macro,
                    innerStrideB, alpha, beta);
            else
                macro_kernel<blockingA, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
        // remainder
        if (blocking_ / 2 >= blocking_micro_ && (i + blocking_ / 2) <= plan->end) {
            if (plan->indexA)
                macro_kernel<blocking_ / 2, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            else if (plan->indexB)
                macro_kernel<blockingA, blocking_ / 2, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            i += blocking_ / 2;
        }
        if (blocking_ / 4 >= blocking_micro_ && (i + blocking_ / 4) <= plan->end) {
            if (plan->indexA)
                macro_kernel<blocking_ / 4, blockingB, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            else if (plan->indexB)
                macro_kernel<blockingA, blocking_ / 4, betaIsZero, floatType, useStreamingStores, conjA>(
                    &A[(i + offDiffAB) * lda], Anext, lda_macro, innerStrideA, &B[i * ldb], Bnext, ldb_macro, innerStrideB, alpha, beta);
            i += blocking_ / 4;
        }
        ptrdiff_t const scalarRemainder = plan->end - i;
        if (scalarRemainder > 0) {
            if (plan->indexA)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, scalarRemainder, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, blockingB, innerStrideB, alpha, beta);
            else if (plan->indexB)
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, blockingA, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, scalarRemainder, innerStrideB, alpha, beta);
            else
                macro_kernel_scalar<betaIsZero, floatType, conjA>(&A[(i + offDiffAB) * lda], lda_macro, blockingA, innerStrideA,
                                                                  &B[i * ldb], ldb_macro, blockingB, innerStrideB, alpha, beta);
        }
    }
}

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool conjA>
void transpose_int_constStride1(floatType const *A, floatType *B, floatType const alpha, floatType const beta, ComputeNode const *plan) {
    ptrdiff_t const end = plan->end - (plan->inc - 1);
    /// @todo Fix code.
    constexpr ptrdiff_t inc       = 1;
    size_t const        lda       = plan->lda;
    size_t const        ldb       = plan->ldb;
    ptrdiff_t const     offDiffAB = plan->offDiffAB;

    if (plan->next != nullptr) {
        for (ptrdiff_t i = plan->start; i < end; i += inc) {
            // recurse
            transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, conjA>(&A[(i + offDiffAB) * lda], &B[i * ldb], alpha,
                                                                                         beta, plan->next);
        }
    } else if constexpr (!betaIsZero) {
        for (ptrdiff_t i = plan->start; i < end; i += inc) {
            if constexpr (conjA)
                B[i * ldb] = alpha * conj(A[(i + offDiffAB) * lda]) + beta * B[i * ldb];
            else
                B[i * ldb] = alpha * A[(i + offDiffAB) * lda] + beta * B[i * ldb];
        }
    } else {
        if constexpr (useStreamingStores) {
            if constexpr (conjA) {
#pragma vector nontemporal
                for (ptrdiff_t i = plan->start; i < end; i += inc) {
                    B[i * ldb] = alpha * conj(A[(i + offDiffAB) * lda]);
                }
            } else {
#pragma vector nontemporal
                for (ptrdiff_t i = plan->start; i < end; i += inc) {
                    B[i * ldb] = alpha * A[(i + offDiffAB) * lda];
                }
            }
        } else if constexpr (conjA) {
            for (ptrdiff_t i = plan->start; i < end; i += inc) {
                B[i * ldb] = alpha * conj(A[(i + offDiffAB) * lda]);
            }
        } else {
            for (ptrdiff_t i = plan->start; i < end; i += inc) {
                B[i * ldb] = alpha * A[(i + offDiffAB) * lda];
            }
        }
    }
}

template <typename floatType>
Transpose<floatType>::Transpose(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA, size_t const innerStrideB,
                                int const dim, floatType const *A, floatType const alpha, floatType *B, floatType const beta,
                                SelectionMethod const selectionMethod, int const numThreads, int const *threadIds, bool const useRowMajor)
    : A_(A), B_(B), alpha_(alpha), beta_(beta), dim_(-1), innerStrideA_(0), innerStrideB_(0), numThreads_(numThreads), masterPlan_(nullptr),
      selectionMethod_(selectionMethod), maxAutotuningCandidates_(-1), selectedParallelStrategyId_(-1), selectedLoopOrderId_(-1),
      conjA_(false) {
#ifdef _OPENMP
    omp_init_lock(&writelock);
#endif

    std::vector<int>    tmpPerm(dim);
    std::vector<size_t> tmpSizeA(dim), tmpOuterSizeA(dim), tmpOuterSizeB(dim), tmpOffsetA(dim), tmpOffsetB(dim);

    accountForRowMajor(sizeA, outerSizeA, outerSizeB, offsetA, offsetB, perm, tmpSizeA.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(),
                       tmpOffsetA.data(), tmpOffsetB.data(), tmpPerm.data(), dim, useRowMajor);

    sizeA_.resize(dim);
    perm_.resize(dim);
    outerSizeA_.resize(dim);
    outerSizeB_.resize(dim);
    offsetA_.resize(dim);
    offsetB_.resize(dim);
    lda_.resize(dim);
    ldb_.resize(dim);
    threadIds_.reserve(dim);
    if (threadIds) {
        // compact threadIds. E.g., 1, 7, 5 -> local_id(1) = 0, local_id(7) = 2,
        // local_id(5) = 1
        for (int i = 0; i < numThreads; ++i)
            threadIds_.push_back(threadIds[i]);
        std::sort(threadIds_.begin(), threadIds_.end());
    } else {
        for (int i = 0; i < numThreads; ++i)
            threadIds_.push_back(i);
    }

    verifyParameter(tmpSizeA.data(), tmpPerm.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(),
                    innerStrideA, innerStrideB, dim);

    innerStrideA_ = innerStrideA;
    innerStrideB_ = innerStrideB;

    // initializes dim_, outerSizeA, outerSizeB, sizeA and perm
    skipIndices(tmpSizeA.data(), tmpPerm.data(), tmpOuterSizeA.data(), tmpOuterSizeB.data(), tmpOffsetA.data(), tmpOffsetB.data(), dim);
    fuseIndices();

    // initializes lda_ and ldb_
    computeLeadingDimensions();

    // create plan
    this->createPlan();
}

template <typename floatType>
Transpose<floatType>::Transpose(Transpose<floatType> const &other)
    : A_(other.A_), B_(other.B_), alpha_(other.alpha_), beta_(other.beta_), dim_(other.dim_), numThreads_(other.numThreads_),
      masterPlan_(other.masterPlan_), selectionMethod_(other.selectionMethod_),
      selectedParallelStrategyId_(other.selectedParallelStrategyId_), selectedLoopOrderId_(other.selectedLoopOrderId_),
      maxAutotuningCandidates_(other.maxAutotuningCandidates_), sizeA_(other.sizeA_), perm_(other.perm_), outerSizeA_(other.outerSizeA_),
      outerSizeB_(other.outerSizeB_), offsetA_(other.offsetA_), offsetB_(other.offsetB_), innerStrideA_(other.innerStrideA_),
      innerStrideB_(other.innerStrideB_), lda_(other.lda_), ldb_(other.ldb_), threadIds_(other.threadIds_), conjA_(other.conjA_) {
#ifdef _OPENMP
    omp_init_lock(&writelock);
#endif
}

template <typename floatType>
Transpose<floatType>::~Transpose() {
#ifdef _OPENMP
    omp_destroy_lock(&writelock);
#endif
}

template <typename floatType>
void Transpose<floatType>::executeEstimate(Plan const *plan) noexcept {
    if (plan == nullptr) {
        fprintf(stderr, "[HPTT] ERROR: plan has not yet been created.\n");
        exit(-1);
    }

    constexpr bool useStreamingStores = false;

    int const numTasks = plan->getNumTasks();
#ifdef _OPENMP
#    pragma omp parallel for num_threads(numThreads_) if (numThreads_ > 1)
#endif
    for (int taskId = 0; taskId < numTasks; taskId++)
        if (perm_[0] != 0) {
            auto rootNode = plan->getRootNode(taskId);
            if (std::abs(beta_) < getZeroThreshold<floatType>()) {
                if (conjA_)
                    transpose_int<blocking_, blocking_, 1, floatType, useStreamingStores, true>(A_, A_, innerStrideA_, B_, B_,
                                                                                                innerStrideB_, 0.0, 1.0, rootNode);
                else
                    transpose_int<blocking_, blocking_, 1, floatType, useStreamingStores, false>(A_, A_, innerStrideA_, B_, B_,
                                                                                                 innerStrideB_, 0.0, 1.0, rootNode);
            } else {
                if (conjA_)
                    transpose_int<blocking_, blocking_, 0, floatType, useStreamingStores, true>(A_, A_, innerStrideA_, B_, B_,
                                                                                                innerStrideB_, 0.0, 1.0, rootNode);
                else
                    transpose_int<blocking_, blocking_, 0, floatType, useStreamingStores, false>(A_, A_, innerStrideA_, B_, B_,
                                                                                                 innerStrideB_, 0.0, 1.0, rootNode);
            }
        } else {
            auto rootNode = plan->getRootNode(taskId);
            if (std::abs(beta_) < getZeroThreshold<floatType>()) {
                if (conjA_)
                    transpose_int_constStride1<1, floatType, useStreamingStores, true>(A_, B_, 0.0, 1.0, rootNode);
                else
                    transpose_int_constStride1<1, floatType, useStreamingStores, false>(A_, B_, 0.0, 1.0, rootNode);
            } else {
                if (conjA_)
                    transpose_int_constStride1<0, floatType, useStreamingStores, true>(A_, B_, 0.0, 1.0, rootNode);
                else
                    transpose_int_constStride1<0, floatType, useStreamingStores, false>(A_, B_, 0.0, 1.0, rootNode);
            }
        }
}

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_1D(floatType const *A, floatType *B, size_t const myStart, size_t const myEnd, ptrdiff_t const offDiffAB_,
                    size_t const lda, size_t const ldb, floatType const alpha, floatType const beta, int numThreads) {
    if constexpr (!betaIsZero) {
        HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if (conjA) B[i * ldb] =
                                         alpha * conj(A[(i + offDiffAB_) * lda]) + beta * B[i * ldb];
                       else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda] + beta * B[i * ldb];)
    } else {
        if constexpr (useStreamingStores)
            HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if constexpr (conjA) B[i * ldb] =
                                             alpha * conj(A[(i + offDiffAB_) * lda]);
                           else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda];)
        else
            HPTT_DUPLICATE(spawnThreads, for (size_t i = myStart; i < myEnd; i++) if constexpr (conjA) B[i * ldb] =
                                             alpha * conj(A[(i + offDiffAB_) * lda]);
                           else B[i * ldb] = alpha * A[(i + offDiffAB_) * lda];)
    }
}

template <bool betaIsZero, typename floatType, bool useStreamingStores, bool spawnThreads, bool conjA>
static void axpy_2D(floatType const *A, size_t const (&lda)[2], floatType *B, size_t const (&ldb)[2], size_t const n0, size_t const myStart,
                    size_t const myEnd, ptrdiff_t const offDiffAB_[2], size_t const offsetB_, floatType const alpha, floatType const beta,
                    int numThreads) {
    if constexpr (!betaIsZero) {
        HPTT_DUPLICATE(spawnThreads,
                       for (size_t j = myStart; j < myEnd; j++) for (size_t i = offsetB_; i < n0 + offsetB_; i++) if constexpr (conjA)
                           B[(i * ldb[0]) + j * ldb[1]] = alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]) +
                                                          beta * B[(i * ldb[0]) + j * ldb[1]];
                       else B[(i * ldb[0]) + j * ldb[1]] =
                           alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]] + beta * B[(i * ldb[0]) + j * ldb[1]];)
    } else {
        if constexpr (useStreamingStores)
            HPTT_DUPLICATE(spawnThreads, for (size_t j = myStart; j < myEnd; j++)
                                             _Pragma("vector nontemporal") for (size_t i = offsetB_; i < n0 + offsetB_;
                                                                                i++) if constexpr (conjA) B[(i * ldb[0]) + j * ldb[1]] =
                                                 alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]);
                           else B[(i * ldb[0]) + j * ldb[1]] = alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]];)
        else
            HPTT_DUPLICATE(spawnThreads, for (size_t j = myStart; j < myEnd; j++) for (size_t i = offsetB_; i < n0 + offsetB_;
                                                                                       i++) if (conjA) B[(i * ldb[0]) + j * ldb[1]] =
                                             alpha * conj(A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]]);
                           else B[(i * ldb[0]) + j * ldb[1]] = alpha * A[((i + offDiffAB_[0]) * lda[0]) + (j + offDiffAB_[1]) * lda[1]];)
    }
}

template <typename floatType>
template <bool spawnThreads>
void Transpose<floatType>::getStartEnd(size_t n, size_t &myStart, size_t &myEnd) const {
#ifdef _OPENMP
    int myLocalThreadId = getLocalThreadId(omp_get_thread_num());
#else
    int myLocalThreadId = 0;
#endif

    if (myLocalThreadId == -1) // skip those threads which do not participate in this plan
    {
        myStart = n;
        myEnd   = n;
        return;
    }
    if constexpr (spawnThreads) { // worksharing will be handled by the OpenMP runtime
        myStart = 0;
        myEnd   = n;
        return;
    }

    size_t const workPerThread = (n + numThreads_ - 1) / numThreads_;
    myStart                    = std::min(n, myLocalThreadId * workPerThread);
    myEnd                      = std::min(n, (myLocalThreadId + 1) * workPerThread);
}

template <typename floatType>
int Transpose<floatType>::getLocalThreadId(int myThreadId) const {
    int myLocalId = -1;
    for (int i = 0; i < numThreads_; ++i)
        if (myThreadId == threadIds_[i])
            myLocalId = i;
    return myLocalId;
}

template <typename floatType>
template <bool useStreamingStores, bool spawnThreads, bool betaIsZero>
void Transpose<floatType>::execute_expert() noexcept {
    if (masterPlan_ == nullptr) {
        fprintf(stderr, "[HPTT] ERROR: master plan has not yet been created.\n");
        exit(-1);
    }

    size_t myStart = 0;
    size_t myEnd   = 0;

    if (dim_ == 1) {
        getStartEnd<spawnThreads>(sizeA_[0], myStart, myEnd);
        ptrdiff_t const offDiffAB_ = (ptrdiff_t)offsetA_[0] - (ptrdiff_t)offsetB_[0];
        if (conjA_)
            axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>(
                A_, B_, myStart + offsetB_[0], myEnd + offsetB_[0], offDiffAB_, lda_[0], ldb_[0], alpha_, beta_, numThreads_);
        else
            axpy_1D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>(
                A_, B_, myStart + offsetB_[0], myEnd + offsetB_[0], offDiffAB_, lda_[0], ldb_[0], alpha_, beta_, numThreads_);
        return;
    } else if (dim_ == 2 && perm_[0] == 0) {
        getStartEnd<spawnThreads>(sizeA_[1], myStart, myEnd);
        ptrdiff_t const offDiffAB_[2] = {((ptrdiff_t)offsetA_[0] - (ptrdiff_t)offsetB_[0]),
                                         ((ptrdiff_t)offsetA_[1] - (ptrdiff_t)offsetB_[1])};
        if (conjA_)
            axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, true>(A_, {lda_[0], lda_[1]}, B_, {ldb_[0], ldb_[1]},
                                                                                   sizeA_[0], myStart + offsetB_[1], myEnd + offsetB_[1],
                                                                                   offDiffAB_, offsetB_[0], alpha_, beta_, numThreads_);
        else
            axpy_2D<betaIsZero, floatType, useStreamingStores, spawnThreads, false>(A_, {lda_[0], lda_[1]}, B_, {ldb_[0], ldb_[1]},
                                                                                    sizeA_[0], myStart + offsetB_[1], myEnd + offsetB_[1],
                                                                                    offDiffAB_, offsetB_[0], alpha_, beta_, numThreads_);
        return;
    }

    int const numTasks   = masterPlan_->getNumTasks();
    int const numThreads = numThreads_;
    getStartEnd<spawnThreads>(numTasks, myStart, myEnd);

    HPTT_DUPLICATE(
        spawnThreads,
        for (int taskId = myStart; taskId < myEnd; taskId++) if (perm_[0] != 0) {
            auto rootNode = masterPlan_->getRootNode(taskId);
            if (conjA_)
                transpose_int<blocking_, blocking_, betaIsZero, floatType, useStreamingStores, true>(
                    A_, A_, innerStrideA_, B_, B_, innerStrideB_, alpha_, beta_, rootNode);
            else
                transpose_int<blocking_, blocking_, betaIsZero, floatType, useStreamingStores, false>(
                    A_, A_, innerStrideA_, B_, B_, innerStrideB_, alpha_, beta_, rootNode);
        } else {
            auto rootNode = masterPlan_->getRootNode(taskId);
            if (conjA_)
                transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, true>(A_, B_, alpha_, beta_, rootNode);
            else
                transpose_int_constStride1<betaIsZero, floatType, useStreamingStores, false>(A_, B_, alpha_, beta_, rootNode);
        })
}
template <typename floatType>
void Transpose<floatType>::execute() noexcept {
    if (masterPlan_ == nullptr) {
        fprintf(stderr, "[HPTT] ERROR: master plan has not yet been created.\n");
        exit(-1);
    }

    bool           spawnThreads       = numThreads_ > 1;
    bool           betaIsZero         = (beta_ == (floatType)0.0);
    constexpr bool useStreamingStores = true;
    if (spawnThreads) {
        if (betaIsZero) {
            this->execute_expert<useStreamingStores, true, true>();
        } else {
            this->execute_expert<useStreamingStores, true, false>();
        }
    } else {
        if (betaIsZero) {
            this->execute_expert<useStreamingStores, false, true>();
        } else {
            this->execute_expert<useStreamingStores, false, false>();
        }
    }
}

template <typename floatType>
void Transpose<floatType>::print() noexcept {
    masterPlan_->print();
}

template <typename floatType>
size_t Transpose<floatType>::getIncrement(int loopIdx) const {
    size_t inc = 1;
    if (perm_[0] != 0) {
        if (loopIdx == 0 || loopIdx == perm_[0])
            inc = blocking_;
    }
    return inc;
}

template <typename floatType>
void Transpose<floatType>::getAvailableParallelism(std::vector<int> &numTasksPerLoop) const {
    numTasksPerLoop.resize(dim_);
    for (int loopIdx = 0; loopIdx < dim_; ++loopIdx) {
        size_t inc               = this->getIncrement(loopIdx);
        numTasksPerLoop[loopIdx] = (sizeA_[loopIdx] + inc - 1) / inc;
    }
}

template <typename floatType>
void Transpose<floatType>::getAllParallelismStrategies(std::list<int> &primeFactorsToMatch, std::vector<int> &availableParallelismAtLoop,
                                                       std::vector<int>              &achievedParallelismAtLoop,
                                                       std::vector<std::vector<int>> &parallelismStrategies) const {
    if (primeFactorsToMatch.size() > 0) {
        // match every primefactor ...
        for (auto p : primeFactorsToMatch) {
            // ... with every loop
            for (int i = 0; i < dim_; i++) {
                std::list<int>   primeFactorsToMatch_(primeFactorsToMatch);
                std::vector<int> availableParallelismAtLoop_(availableParallelismAtLoop);
                std::vector<int> achievedParallelismAtLoop_(achievedParallelismAtLoop);

                primeFactorsToMatch_.erase(std::find(primeFactorsToMatch_.begin(), primeFactorsToMatch_.end(), p));
                availableParallelismAtLoop_[i] = (availableParallelismAtLoop_[i] + p - 1) / p;
                achievedParallelismAtLoop_[i] *= p;

                this->getAllParallelismStrategies(primeFactorsToMatch_, availableParallelismAtLoop_, achievedParallelismAtLoop_,
                                                  parallelismStrategies);
            }
        }
    } else {
        // avoid duplicates
        if (parallelismStrategies.end() == std::find(parallelismStrategies.begin(), parallelismStrategies.end(), achievedParallelismAtLoop))
            parallelismStrategies.push_back(achievedParallelismAtLoop);
    }
}

// balancing if one tries to parallelize avail many tasks with req many threads
// e.g., balancing(3,4) = 0.75
static float getBalancing(int avail, int req) {
    return ((float)(avail)) / (float)((int)((avail + req - 1) / req) * req);
}

template <typename floatType>
float Transpose<floatType>::getLoadBalance(std::vector<int> const &parallelismStrategy) const {
    float load_balance = 1.0;
    int   totalTasks   = 1;
    for (int i = 0; i < dim_; ++i) {

        size_t inc = this->getIncrement(i);
        while (sizeA_[i] < inc)
            inc /= 2;
        size_t availableParallelism = (sizeA_[i] + inc - 1) / inc;

        if (i == 0 || perm_[i] == 0)
            // account for the load-imbalancing due to blocking
            load_balance *= getBalancing(sizeA_[i], inc);
        load_balance *= getBalancing(availableParallelism, parallelismStrategy[i]);
        totalTasks *= parallelismStrategy[i];
    }

    // how well can these tasks be distributed among numThreads_?
    //  e.g., totalTasks = 3, numThreads = 8 => 3./8
    //  e.g., totalTasks = 5, numThreads = 8 => 5./8
    //  e.g., totalTasks = 15, numThreads = 8 => 15./16
    //  e.g., totalTasks = 17, numThreads = 8 => 17./24
    float workDistribution = ((float)totalTasks) / (((totalTasks + numThreads_ - 1) / numThreads_) * numThreads_);

    load_balance *= workDistribution;
    return load_balance;
}

template <typename floatType>
void Transpose<floatType>::getBestParallelismStrategy(std::vector<int> &bestParallelismStrategy) const {
    std::vector<int> availableParallelismAtLoop;
    this->getAvailableParallelism(availableParallelismAtLoop);
    int totalAvailableParallelism =
        std::accumulate(availableParallelismAtLoop.begin(), availableParallelismAtLoop.end(), 1, std::multiplies<int>());

    // reduce the probability of parallelizing the stride-1 index
    // if this loops would be parallelized, these two statements ensure that each
    // thread would have at least two macro-kernels of work at this loop-level
    //
    // However, if the total available parallelism is too small, then we do not
    // artificially limit the available parallelism further
    int reduceParallelismB = 4; // avoid parallelization in stride-1 B more strongly
    int reduceParallelismA = 2;
    if (totalAvailableParallelism < 2 * numThreads_)
        reduceParallelismB = 1;
    else if (totalAvailableParallelism < 4 * numThreads_)
        reduceParallelismB = 2;
    totalAvailableParallelism =
        (totalAvailableParallelism / availableParallelismAtLoop[perm_[0]]) * (availableParallelismAtLoop[perm_[0]] / reduceParallelismB);
    if (totalAvailableParallelism < 2 * numThreads_)
        reduceParallelismA = 1;
    availableParallelismAtLoop[perm_[0]] = std::max(1, availableParallelismAtLoop[perm_[0]] / reduceParallelismB);
    availableParallelismAtLoop[0]        = std::max(1, availableParallelismAtLoop[0] / reduceParallelismA);

    // Objectives: 1) load-balancing
    //             2) avoid parallelizing stride-1 loops (rational: less
    //             consecutive memory accesses) 3) avoid false sharing

    std::vector<int> loopsAllowed;
    for (int i = dim_ - 1; i >= 1; i--)
        if (perm_[i] != 0)
            loopsAllowed.push_back(perm_[i]);
    std::vector<int> loopsAllowedStride1{0, perm_[0]};

    int            totalTasks = 1; // goal: totalTasks should be a close multiple of numTasks_
    std::list<int> primeFactors;
    getPrimeFactors(numThreads_, primeFactors);

    // 1. parallelize using 100% load balancing
    parallelize(bestParallelismStrategy, availableParallelismAtLoop, totalTasks, primeFactors, 1.0, loopsAllowed);

    if (totalTasks != numThreads_) { // no perfect match has been found

        // Option 1: keep parallelizing non-stride-1 loops only, but allowing
        // load-imbalance
        std::vector<int> strat1(bestParallelismStrategy);
        std::vector<int> avail1(availableParallelismAtLoop);
        std::list<int>   primes1(primeFactors);
        int              totalTasks1 = totalTasks;
        parallelize(strat1, avail1, totalTasks1, primes1, 0.92, loopsAllowed);
        if (getLoadBalance(strat1) > 0.90) {
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
            return;
        }

        if (perm_[0] != 0) {
            // Option 2: also parallelize stride-1 loops, enforcing perfect loop
            // balancing
            std::vector<int> strat2(bestParallelismStrategy);
            std::vector<int> avail2(availableParallelismAtLoop);
            std::list<int>   primes2(primeFactors);
            int              totalTasks2 = totalTasks;
            parallelize(strat2, avail2, totalTasks2, primes2, 1.0, loopsAllowedStride1);
            if (getLoadBalance(strat2) > 0.92) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            // keep on going based on strat1
            parallelize(strat1, avail1, totalTasks1, primes1, 1.0, loopsAllowedStride1);
            if (getLoadBalance(strat1) > 0.90) {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }

            // keep on going based on strat2
            parallelize(strat2, avail2, totalTasks2, primes2, 0.92, loopsAllowed);
            if (getLoadBalance(strat2) > 0.92) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            if (getLoadBalance(strat1) > 0.80) // reduced threshold
            {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }
            if (getLoadBalance(strat2) > 0.82) // reduced threshold
            {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            parallelize(strat1, avail1, totalTasks1, primes1, 0.9, loopsAllowedStride1);
            parallelize(strat2, avail2, totalTasks2, primes2, 0.8, loopsAllowed);
            float lb1 = getLoadBalance(strat1);
            float lb2 = getLoadBalance(strat2);
            //         printVector(strat2,"strat2");
            //         printf("strat2: %f\n",getLoadBalance(strat2));
            if (lb1 > 0.8 && lb2 < 0.85 || lb1 > lb2 && lb1 > 0.75) {
                std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
                return;
            }
            if (lb2 >= 0.85) {
                std::copy(strat2.begin(), strat2.end(), bestParallelismStrategy.begin());
                return;
            }

            // fallback
            std::vector<int> allLoops;
            for (int i = dim_ - 1; i >= 1; i--)
                allLoops.push_back(perm_[i]);
            allLoops.push_back(0);
            allLoops.push_back(perm_[0]);
            parallelize(strat1, avail1, totalTasks1, primes1, 0., allLoops);
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());

        } else {
            parallelize(strat1, avail1, totalTasks1, primes1, 0.0, loopsAllowed);
            std::copy(strat1.begin(), strat1.end(), bestParallelismStrategy.begin());
        }
    }
}

template <typename floatType>
void Transpose<floatType>::parallelize(std::vector<int> &parallelismStrategy, std::vector<int> &availableParallelismAtLoop, int &totalTasks,
                                       std::list<int> &primeFactors, float const minBalancing, std::vector<int> const &loopsAllowed) const

{
    bool suboptimalParallelizationUsed = false;
    // find loop which minimizes load imbalance for the given prime factor
    for (auto it = primeFactors.begin(); it != primeFactors.end(); it++) {
        int   suitedLoop    = -1;
        float bestBalancing = 0;

        for (auto idx : loopsAllowed) {
            float balancing = getBalancing(availableParallelismAtLoop[idx], *it);
            if (balancing > bestBalancing) {
                bestBalancing = balancing;
                suitedLoop    = idx;
            }
        }
        // allow up to one slightly less optimal splitting to prefer parallelizing
        // idx=0 over idx=perm[0]
        if (suboptimalParallelizationUsed == false && suitedLoop == perm_[0] && getBalancing(availableParallelismAtLoop[0], *it) >= 0.949) {
            suitedLoop                    = 0;
            suboptimalParallelizationUsed = true;
        }
        if (suitedLoop != -1 && bestBalancing >= minBalancing) {
            availableParallelismAtLoop[suitedLoop] /= *it;
            parallelismStrategy[suitedLoop] *= *it;
            totalTasks *= *it;
            it = primeFactors.erase(it);
            it--;
        }
    }
}

template <typename floatType>
double Transpose<floatType>::parallelismCostHeuristic(std::vector<int> const &achievedParallelismAtLoop) const {
    std::vector<int> availableParallelismAtLoop;
    this->getAvailableParallelism(availableParallelismAtLoop);

    double cost = 1;
    // penalize load-imbalance
    for (int loopIdx = 0; loopIdx < dim_; ++loopIdx) {
        if (achievedParallelismAtLoop[loopIdx] <= 1)
            continue;

        int const blocksPerThread =
            (availableParallelismAtLoop[loopIdx] + achievedParallelismAtLoop[loopIdx] - 1) / achievedParallelismAtLoop[loopIdx];
        int       inc           = this->getIncrement(loopIdx);
        int const effectiveSize = blocksPerThread * inc * achievedParallelismAtLoop[loopIdx];
        cost *= ((double)(effectiveSize) / sizeA_[loopIdx]);
    }

    // penalize parallelization of stride-1 loops
    if (perm_[0] == 0)
        cost *= std::pow(1.01, achievedParallelismAtLoop[0] - 1); // strongly penalize this case

    cost *= std::pow(1.00010, std::min(16, achievedParallelismAtLoop[0] - 1));        // if at all, prefer ...
    cost *= std::pow(1.00015, std::min(16, achievedParallelismAtLoop[perm_[0]] - 1)); // parallelization in stride-1 of A

    int const workPerThread =
        (availableParallelismAtLoop[perm_[0]] + achievedParallelismAtLoop[perm_[0]] - 1) / achievedParallelismAtLoop[perm_[0]];
    if (workPerThread * sizeof(floatType) % 64 != 0 && achievedParallelismAtLoop[perm_[0]] > 1) { // avoid false-sharing
        cost *= std::pow(1.00015, std::min(16, achievedParallelismAtLoop[perm_[0]] - 1));         // penalize this parallelization again
    }
    return cost;
}

template <typename floatType>
void Transpose<floatType>::getParallelismStrategies(std::vector<std::vector<int>> &parallelismStrategies) const {
    parallelismStrategies.clear();
    if (numThreads_ == 1) {
        parallelismStrategies.emplace_back(std::vector<int>(dim_, 1));
        return;
    }
    std::vector<int> bestParallelismStrategy(dim_, 1);
    getBestParallelismStrategy(bestParallelismStrategy);
    if (this->infoLevel_ > 0)
        printf("Loadbalancing: %f\n", getLoadBalance(bestParallelismStrategy));

    if (selectionMethod_ == ESTIMATE) {
        parallelismStrategies.push_back(bestParallelismStrategy);
        return;
    }

    // ATTENTION: we don't care about the case where numThreads_ is a large prime
    // number... (sorry, KNC)
    //
    // we factorize numThreads into its prime factors because we have to match
    // every one to a certain loop. In principle every loop could be used to
    // match every primefactor, but some choices are preferable over others.
    // E.g., we want to achieve good load-balancing _and_ try to avoid the
    // stride-1 index of B (due to false sharing)
    std::list<int> primeFactors;
    getPrimeFactors(numThreads_, primeFactors);
    if (this->infoLevel_ > 0)
        printVector(primeFactors, "primes");

    std::vector<int> availableParallelismAtLoop;
    this->getAvailableParallelism(availableParallelismAtLoop);
    if (this->infoLevel_ > 0)
        printVector(availableParallelismAtLoop, "available Parallelism");

    std::vector<int> achievedParallelismAtLoop(dim_, 1);

    this->getAllParallelismStrategies(primeFactors, availableParallelismAtLoop, achievedParallelismAtLoop, parallelismStrategies);

    // sort according to loop heuristic
    std::sort(parallelismStrategies.begin(), parallelismStrategies.end(),
              [this](std::vector<int> const loopOrder1, std::vector<int> const loopOrder2) {
                  return this->parallelismCostHeuristic(loopOrder1) < this->parallelismCostHeuristic(loopOrder2);
              });

    parallelismStrategies.insert(parallelismStrategies.begin(), bestParallelismStrategy);

    if (this->infoLevel_ > 1)
        for (auto strat : parallelismStrategies) {
            printVector(strat, "parallelization");
            printf("cost: %f\n", this->parallelismCostHeuristic(strat));
        }
}

template <typename floatType>
void Transpose<floatType>::verifyParameter(size_t const *size, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                           size_t const *offsetA, size_t const *offsetB, size_t const innerStrideA,
                                           size_t const innerStrideB, int const dim) const {
    if (dim < 1) {
        fprintf(stderr, "[HPTT] ERROR: dimensionality too low.\n");
        exit(-1);
    }

    std::vector<int> found(dim, 0);

    for (int i = 0; i < dim; ++i) {
        if (size[i] <= 0) {
            fprintf(stderr, "[HPTT] ERROR: size at position %d is invalid\n", i);
            exit(-1);
        }
        found[perm[i]] = 1;
    }

    for (int i = 0; i < dim; ++i)
        if (found[i] <= 0) {
            fprintf(stderr, "[HPTT] ERROR: permutation invalid\n");
            exit(-1);
        }

    if (outerSizeA != NULL)
        for (int i = 0; i < dim; ++i)
            if (outerSizeA[i] < size[i]) {
                fprintf(stderr, "[HPTT] ERROR: outerSizeA invalid\n");
                exit(-1);
            }

    if (outerSizeB != NULL)
        for (int i = 0; i < dim; ++i)
            if (outerSizeB[i] < size[perm[i]]) {
                fprintf(stderr, "[HPTT] ERROR: outerSizeB invalid\n");
                exit(-1);
            }

    if (offsetA != NULL)
        for (int i = 0; i < dim; ++i)
            if (offsetA[i] + size[i] > outerSizeA[i]) {
                fprintf(stderr, "[HPTT] ERROR: offsetA invalid\n");
                exit(-1);
            }

    if (offsetB != NULL)
        for (int i = 0; i < dim; ++i)
            if (offsetB[i] + size[perm[i]] > outerSizeB[i]) {
                fprintf(stderr, "[HPTT] ERROR: offsetB invalid\n");
                exit(-1);
            }

    if (innerStrideA < 0) {
        fprintf(stderr, "[HPTT] ERROR: innerStrideA invalid\n");
        exit(-1);
    }

    if (innerStrideB < 0) {
        fprintf(stderr, "[HPTT] ERROR: innerStrideB invalid\n");
        exit(-1);
    }
}

template <typename floatType>
void Transpose<floatType>::computeLeadingDimensions() {
    lda_[0] = innerStrideA_;
    if (outerSizeA_[0] == -1)
        for (int i = 1; i < dim_; ++i)
            lda_[i] = lda_[i - 1] * sizeA_[i - 1];
    else
        for (int i = 1; i < dim_; ++i)
            lda_[i] = outerSizeA_[i - 1] * lda_[i - 1];

    ldb_[0] = innerStrideB_;
    if (outerSizeB_[0] == -1)
        for (int i = 1; i < dim_; ++i)
            ldb_[i] = ldb_[i - 1] * sizeA_[perm_[i - 1]];
    else
        for (int i = 1; i < dim_; ++i)
            ldb_[i] = outerSizeB_[i - 1] * ldb_[i - 1];
}

template <typename floatType>
void Transpose<floatType>::skipIndices(size_t const *sizeA, int const *perm, size_t const *outerSizeA, size_t const *outerSizeB,
                                       size_t const *offsetA, size_t const *offsetB, int const dim) {
    for (int i = 0; i < dim; ++i) {
        perm_[i]  = perm[i];
        sizeA_[i] = sizeA[i];
        if (outerSizeA)
            outerSizeA_[i] = outerSizeA[i];
        else
            outerSizeA_[i] = sizeA[i];
        if (outerSizeB)
            outerSizeB_[i] = outerSizeB[i];
        else
            outerSizeB_[i] = sizeA[perm[i]];
        if (offsetA)
            offsetA_[i] = offsetA[i];
        else
            offsetA_[i] = 0;
        if (offsetB)
            offsetB_[i] = offsetB[i];
        else
            offsetB_[i] = 0;
    }

    size_t skipped = 0;
    for (int i = 0; i < dim; ++i) {
        int idxB = 0;
        for (; idxB < dim; ++idxB)
            if (perm[idxB] == i)
                break;
        if (sizeA[i] == 1 && (!outerSizeA || outerSizeA[i] == 1) && (!outerSizeB || outerSizeB[idxB] == 1)) {
            sizeA_[i]         = -1;
            outerSizeA_[i]    = -1;
            outerSizeB_[idxB] = -1;
            offsetA_[i]       = -1;
            offsetB_[idxB]    = -1;
            perm_[idxB]       = -1;
            skipped++;
        }
    }
    // compact arrays (remove -1)
    for (int i = 0; i < dim; ++i)
        if (sizeA_[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (sizeA_[j] != -1)
                    break;
            if (j < dim)
                std::swap(sizeA_[i], sizeA_[j]);
        }
    for (int i = 0; i < dim; ++i)
        if (outerSizeA_[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (outerSizeA_[j] != -1)
                    break;
            if (j < dim) {
                std::swap(outerSizeA_[i], outerSizeA_[j]);
                std::swap(offsetA_[i], offsetA_[j]);
            }
        }
    for (int i = 0; i < dim; ++i)
        if (outerSizeB_[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (outerSizeB_[j] != -1)
                    break;
            if (j < dim) {
                std::swap(outerSizeB_[i], outerSizeB_[j]);
                std::swap(offsetB_[i], offsetB_[j]);
            }
        }
    for (int i = 0; i < dim; ++i)
        if (perm_[i] == -1) {
            int j = i + 1;
            for (; j < dim; ++j)
                if (perm_[j] != -1)
                    break;
            if (j < dim)
                std::swap(perm_[i], perm_[j]);
        }

    dim_ = dim - skipped;
    if (dim_ == 0) {
        dim_ = 1;
        perm_.resize(dim_);
        sizeA_.resize(dim_);
        outerSizeA_.resize(dim_);
        outerSizeB_.resize(dim_);
        perm_[0]       = 0;
        sizeA_[0]      = 1;
        outerSizeA_[0] = 1;
        outerSizeB_[0] = 1;
        offsetA_[0]    = 0;
        offsetB_[0]    = 0;
    } else {
        perm_.resize(dim_);
        sizeA_.resize(dim_);
        outerSizeA_.resize(dim_);
        outerSizeB_.resize(dim_);
        offsetA_.resize(dim_);
        offsetB_.resize(dim_);

        // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
        int currentValue = 0;
        for (int i = 0; i < dim_; ++i) {
            // find smallest element in perm_ and rename it to currentValue
            int minValue = std::numeric_limits<int>::max();
            int minPos   = -1;
            for (int pos = 0; pos < dim_; ++pos) {
                if (perm_[pos] >= currentValue && perm_[pos] < minValue) {
                    minValue = perm_[pos];
                    minPos   = pos;
                }
            }
            perm_[minPos] = currentValue; // minValue renamed to currentValue
            currentValue++;
        }
    }

#ifdef DEBUG
    printVector(perm_, "perm");
    printVector(sizeA_, "sizeA");
    printVector(outerSizeA_, "outerSizeA");
    printVector(outerSizeB_, "outerSizeB");
    printVector(offsetA_, "offsetA");
    printVector(offsetB_, "offsetB");
    printf("dim: %d\n", dim_);
    printf("alpha: %f\n", alpha_);
    printf("beta: %f\n", beta_);
    printf("innerStrideA: %lu\n", innerStrideA_);
    printf("innerStrideB: %lu\n", innerStrideB_);
#endif
}

/**
 * \brief fuses indices whenever possible
 * \detailed For instance:
 *           perm=3,1,2,0 & size=10,11,12,13  becomes: perm=2,1,0 &
 * size=10,11*12,13 \return This function will initialize sizeA_, perm_,
 * outerSizeA_, outersize_ and dim_
 */
template <typename floatType>
void Transpose<floatType>::fuseIndices() {
    std::list<std::tuple<int, int>> fusedIndices;

    std::vector<int> perm;
    // correct perm
    for (int i = 0; i < dim_; ++i) {
        // merge indices if the two consecutive entries are identical
        int toMerge = i;
        perm.push_back(perm_[i]);
        /* By definition if size == outerSize, then no offsets are present. However,
         *  by merging with the subsequent dimension the stride the offset depends upon
         *  is lost. Therefore, the offset of the next offset must be zero too! */
        while (i + 1 < dim_ && perm_[i] + 1 == perm_[i + 1] && (sizeA_[perm_[i]] == outerSizeA_[perm_[i]]) &&
               (sizeA_[perm_[i]] == outerSizeB_[i]) && (offsetA_[perm_[i + 1]] == 0) && (offsetB_[i + 1] == 0)) {
#ifdef DEBUG
            fprintf(stderr, "[HPTT] MERGING indices %d and %d\n", perm_[i], perm_[i + 1]);
#endif
            fusedIndices.emplace_back(std::make_tuple(perm_[toMerge], perm_[i + 1]));
            i++;
        }
    }

    // correct sizes and outer-sizes
    for (auto tup : fusedIndices) {
        sizeA_[std::get<0>(tup)] *= sizeA_[std::get<1>(tup)];
        outerSizeA_[std::get<0>(tup)] *= outerSizeA_[std::get<1>(tup)];
        outerSizeA_[std::get<1>(tup)] = -1;
        offsetA_[std::get<1>(tup)]    = -1;

        auto pos1 = std::find(perm_.begin(), perm_.end(), std::get<0>(tup)) - perm_.begin();
        auto pos2 = std::find(perm_.begin(), perm_.end(), std::get<1>(tup)) - perm_.begin();
        outerSizeB_[pos1] *= outerSizeB_[pos2];
        outerSizeB_[pos2] = -1;
        offsetB_[pos2]    = -1;
    }

    if (fusedIndices.size() > 0) {
        perm_ = perm;
        // remove gaps in the perm, if requried (e.g., perm=3,1,0 -> 2,1,0)
        int currentValue = 0;
        for (int i = 0; i < perm_.size(); ++i) {
            // find smallest element in perm_ and rename it to currentValue
            int minValue = std::numeric_limits<int>::max();
            int minPos   = -1;
            for (int pos = 0; pos < perm_.size(); ++pos) {
                if (perm_[pos] >= currentValue && perm_[pos] < minValue) {
                    minValue = perm_[pos];
                    minPos   = pos;
                }
            }
#ifdef DEBUG
            printf("perm[%d]: %d -> %d\n", minPos, perm_[minPos], currentValue);
#endif
            perm_[minPos]        = currentValue; // minValue renamed to currentValue
            sizeA_[currentValue] = sizeA_[minValue];
            currentValue++;
        }

        // compact outer size (e.g.: outerSizeA_[] = {24,-1,5,-1,13} ->
        // {24,5,13,-1,-1} -> {24,5,13}
        for (int i = 0; i < dim_; ++i)
            if (outerSizeA_[i] == -1) {
                int j = i + 1;
                for (; j < dim_; ++j)
                    if (outerSizeA_[j] != -1)
                        break;
                if (j < dim_) {
                    std::swap(outerSizeA_[i], outerSizeA_[j]);
                    std::swap(offsetA_[i], offsetA_[j]);
                }
            }
        for (int i = 0; i < dim_; ++i)
            if (outerSizeB_[i] == -1) {
                int j = i + 1;
                for (; j < dim_; ++j)
                    if (outerSizeB_[j] != -1)
                        break;
                if (j < dim_) {
                    std::swap(outerSizeB_[i], outerSizeB_[j]);
                    std::swap(offsetB_[i], offsetB_[j]);
                }
            }
        dim_ -= fusedIndices.size();
        outerSizeA_.resize(dim_);
        outerSizeB_.resize(dim_);
        offsetA_.resize(dim_);
        offsetB_.resize(dim_);
        sizeA_.resize(dim_);
        perm_.resize(dim_);

#ifdef DEBUG
        printf("\nperm_new: ");
        for (int i = 0; i < dim_; ++i)
            printf("%d ", perm_[i]);
        printf("\nsizes_new: ");
        for (int i = 0; i < dim_; ++i)
            printf("%lu ", sizeA_[i]);
        printf("\nouterSizeA_new: ");
        for (int i = 0; i < dim_; ++i)
            printf("%lu ", outerSizeA_[i]);
        printf("\nouterSizeB_new: ");
        for (int i = 0; i < dim_; ++i)
            printf("%lu ", outerSizeB_[i]);
        printf("\noffsetA_new: ");
        for (int i = 0; i < dim_; ++i)
            printf("%lu ", offsetA_[i]);
        printf("\noffsetB_new: ");
        for (int i = 0; i < dim_; ++i)
            printf("%lu ", offsetB_[i]);
        printf("\n");
#endif
    }
}

// returns the best loop order (same as the best one with exhaustive search)
template <typename floatType>
void Transpose<floatType>::getBestLoopOrder(std::vector<int> &loopOrder) const {
    auto totalOuterSizeA = std::accumulate(outerSizeA_.begin(), outerSizeA_.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
    auto totalOuterSizeB = std::accumulate(outerSizeB_.begin(), outerSizeB_.end(), 1, std::multiplies<size_t>()) * sizeof(floatType);
    if (totalOuterSizeA > totalOuterSizeB && totalOuterSizeB <= 22 * 1024. * 1024.) // B is likely to fit into L3 cache
    {
        // prefer accesses to A over those to B (Rationale: reduce TLB misses)
        for (int i = 0; i < dim_; ++i)
            loopOrder[dim_ - 1 - i] = i; // innermost loop idx is stored at dim_-1
        return;
    } else if (totalOuterSizeB > totalOuterSizeA && totalOuterSizeA <= 22 * 1024. * 1024.) // B is likely to fit into L3 cache
    {
        // prefer accesses to B over those to A (Rationale: reduce TLB misses)
        for (int i = 0; i < dim_; ++i)
            loopOrder[dim_ - 1 - i] = dim_ - 1 - i; // innermost loop idx is stored at dim_-1
        return;
    }

    // create cost matrix; cost[i,idx] === cost for idx being at loop-level i
    std::vector<double> costs(dim_ * dim_);
    for (int i = 0; i < dim_; ++i) {
        for (int idx = 0; idx < dim_; ++idx) { // idx is at loop i
            double cost = 0;
            if (i != 0) {
                int const        posB        = findPos(idx, perm_);
                int const        importanceA = (1 << (dim_ - idx));  // stride-1 has the most importance ...
                int const        importanceB = (1 << (dim_ - posB)); // subsequent indices are half as important
                int const        penalty     = 10 * (1 << (i - 1));
                constexpr double bias        = 1.01;
                cost                         = (importanceA + importanceB * bias) * penalty;
            }
            costs[i + idx * dim_] = cost;
        }
    }
    std::list<int> availLoopLevels; // available rows
    std::list<int> availIndices;
    for (int i = 0; i < dim_; ++i) {
        availLoopLevels.push_back(i);
        availIndices.push_back(i);
    }

    // create best loop order constructively without generating all
    for (int i = 0; i < dim_; ++i) {
        // find column with maximum cost
        int    selectedIdx = 0;
        double maxValueAll = 0;
        for (auto c : availIndices) {
            double maxValue = 0;
            for (auto r : availLoopLevels) {
                double const val = costs[c * dim_ + r];
                maxValue         = (val > maxValue) ? val : maxValue;
            }

            if (maxValue > maxValueAll) {
                maxValueAll = maxValue;
                selectedIdx = c;
            }
        }
        // find minimum in that column
        int    selectedLoopLevel = 0;
        double minValue          = 1e100;
        for (auto r : availLoopLevels) {
            double const val = costs[selectedIdx * dim_ + r];
            if (val < minValue) {
                minValue          = val;
                selectedLoopLevel = r;
            }
        }
        // update loop order
        loopOrder[dim_ - 1 - i] = selectedIdx; // innermost loop idx is stored at dim_-1
        // remove selected row
        for (auto it = availLoopLevels.begin(); it != availLoopLevels.end(); it++)
            if (*it == selectedLoopLevel) {
                availLoopLevels.erase(it);
                break;
            }
        // remove selected col
        for (auto it = availIndices.begin(); it != availIndices.end(); it++)
            if (*it == selectedIdx) {
                availIndices.erase(it);
                break;
            }
    }
}

template <typename floatType>
double Transpose<floatType>::loopCostHeuristic(std::vector<int> const &loopOrder) const {
    double loopCost = 0.0;
    for (int i = 1; i < dim_; ++i) {
        int const idx         = loopOrder[dim_ - 1 - i];
        int const posB        = findPos(idx, perm_);
        int const importanceA = (1 << (dim_ - idx));  // stride-1 has the most importance ...
        int const importanceB = (1 << (dim_ - posB)); // subsequent indices are half as important
        int const penalty     = 10 * (1 << (i - 1));
        double    bias        = 1.01;
        loopCost += (importanceA + importanceB * bias) * penalty;
    }

    return loopCost;
}

template <typename floatType>
void Transpose<floatType>::getLoopOrders(std::vector<std::vector<int>> &loopOrders) const {
    loopOrders.clear();
    if (selectionMethod_ == ESTIMATE) {
        loopOrders.emplace_back(std::vector<int>(dim_));
        getBestLoopOrder(loopOrders[0]);
        return;
    }

    std::vector<int> loopOrder;
    for (int i = 0; i < dim_; i++)
        loopOrder.push_back(i);

    // create all loopOrders
    do {
        if (perm_[0] == 0 && loopOrder[dim_ - 1] != 0)
            continue; // ATTENTION: we skip all loop-orders where the stride-1 index
                      // is not the inner-most loop iff perm[0] == 0 (both for perf &
                      // correctness)

        loopOrders.push_back(loopOrder);
    } while (std::next_permutation(loopOrder.begin(), loopOrder.end()));

    // sort according to loop heuristic
    std::sort(loopOrders.begin(), loopOrders.end(), [this](std::vector<int> const loopOrder1, std::vector<int> const loopOrder2) {
        return this->loopCostHeuristic(loopOrder1) < this->loopCostHeuristic(loopOrder2);
    });

    if (this->infoLevel_ > 1)
        for (auto loopOrder : loopOrders) {
            printVector(loopOrder, "loop");
            printf("penalty: %f\n", loopCostHeuristic(loopOrder));
        }
}

template <typename floatType>
void Transpose<floatType>::createPlan() {
//   printf("entering createPlan()\n");
#ifdef HPTT_TIMERS
    double timeStart = omp_get_wtime();
#endif

    std::vector<std::shared_ptr<Plan>> allPlans;
    createPlans(allPlans);

#ifdef HPTT_TIMERS
    printf("createPlans() took %f ms\n", (omp_get_wtime() - timeStart) * 1000);
    timeStart = omp_get_wtime();
#endif
    masterPlan_ = selectPlan(allPlans);
    if (this->infoLevel_ > 0) {
        printf("Configuration of best plan:\n");
        masterPlan_->print();
    }
#ifdef HPTT_TIMERS
    printf("SelectPlan() took %f ms\n", (omp_get_wtime() - timeStart) * 1000);
#endif
}

template <typename floatType>
void Transpose<floatType>::createPlans(std::vector<std::shared_ptr<Plan>> &plans) const {
    if (dim_ == 1 || (dim_ == 2 && perm_[0] == 0)) {
        plans.emplace_back(new Plan); // create dummy plan
        return;                       // handled within execute()
    }
#ifdef HPTT_TIMERS
    double parallelStrategiesTime = omp_get_wtime();
#endif
    std::vector<std::vector<int>> parallelismStrategies;
    this->getParallelismStrategies(parallelismStrategies);
#ifdef HPTT_TIMERS
    printf("There exists %d parallel strategies. Time: %f ms\n", parallelismStrategies.size(),
           (omp_get_wtime() - parallelStrategiesTime) * 1000);

    double loopOrdersTime = omp_get_wtime();
#endif
    std::vector<std::vector<int>> loopOrders;
    this->getLoopOrders(loopOrders);
#ifdef HPTT_TIMERS
    printf("There exists %d loop orders. Time: %f ms\n", loopOrders.size(), (omp_get_wtime() - loopOrdersTime) * 1000);
#endif

    if (selectedParallelStrategyId_ != -1) {
        int              selectedParallelStrategyId = std::min((int)parallelismStrategies.size() - 1, selectedParallelStrategyId_);
        std::vector<int> parStrategy(parallelismStrategies[selectedParallelStrategyId]);
        printVector(parStrategy, "selected parallel: ");
        parallelismStrategies.clear();
        parallelismStrategies.push_back(parStrategy);
    }
    if (selectedLoopOrderId_ != -1) {
        int              selectedLoopOrderId = std::min((int)loopOrders.size() - 1, selectedLoopOrderId_);
        std::vector<int> loopOrder(loopOrders[selectedLoopOrderId]);
        printVector(loopOrder, "selected loopOrder: ");
        loopOrders.clear();
        loopOrders.push_back(loopOrder);
    }

    int const posStride1A_inB = findPos(0, perm_);
    int const posStride1B_inA = perm_[0];

    // combine the loopOrder and parallelismStrategies according to their
    // heuristics, search the space with a growing rectangle (from best to worst,
    // see line marked with ***)
    bool done = false;
    for (int start = 0; start < std::max(parallelismStrategies.size(), loopOrders.size()) && !done; start++)
        for (int i = 0; i < parallelismStrategies.size() && !done; i++) {
            for (int j = 0; j < loopOrders.size() && !done; j++) {
                if (i > start || j > start || (i != start && j != start))
                    continue; // these are already done ***

                auto      numThreadsAtLoop = parallelismStrategies[i];
                auto      loopOrder        = loopOrders[j];
                auto      plan             = std::make_shared<Plan>(loopOrder, numThreadsAtLoop);
                int const numTasks         = plan->getNumTasks();

#ifdef _OPENMP
#    pragma omp parallel for num_threads(numThreads_) if (numThreads_ > 1)
#endif
                for (int taskId = 0; taskId < numTasks; taskId++) {
                    ComputeNode *currentNode = plan->getRootNode(taskId);

                    int numThreadsPerComm = numTasks; // global communicator // e.g., 6
                    int taskIdComm        = taskId;   // e.g., 0,1,2,3,4,5
                    // divide each loop-level l, corresponding to index loopOrder[l], into
                    // numThreadsAtLoop[index] chunks
                    for (int l = 0; l < dim_; ++l) {
                        int const index  = loopOrder[l];
                        currentNode->inc = this->getIncrement(index);

                        int const numTasksAtLevel         = numThreadsAtLoop[index];                                   //  e.g., 3
                        int const numParallelismAvailable = (sizeA_[index] + currentNode->inc - 1) / currentNode->inc; // e.g., 5
                        int const workPerThread = (numParallelismAvailable + numTasksAtLevel - 1) / numTasksAtLevel;   // ceil(5/3) = 2

                        numThreadsPerComm /= numTasksAtLevel;                // numThreads in next communicator // 6/3 = 2
                        int const commId = (taskIdComm / numThreadsPerComm); //  = 0,0,1,1,2,2
                        taskIdComm       = taskIdComm % numThreadsPerComm;   // local taskId in next
                                                                             // communicator // 0,1,0,1,0,1

                        if (index == 0)
                            currentNode->indexA = true;
                        if (findPos(index, perm_) == 0)
                            currentNode->indexB = true;
                        currentNode->start = std::min(sizeA_[index] + offsetB_[findPos(index, perm_)],
                                                      commId * workPerThread * currentNode->inc + offsetB_[findPos(index, perm_)]);
                        currentNode->end   = std::min(sizeA_[index] + offsetB_[findPos(index, perm_)],
                                                      (commId + 1) * workPerThread * currentNode->inc + offsetB_[findPos(index, perm_)]);

                        currentNode->lda       = lda_[index];
                        currentNode->ldb       = ldb_[findPos(index, perm_)];
                        currentNode->offDiffAB = (ptrdiff_t)offsetA_[index] - (ptrdiff_t)offsetB_[findPos(index, perm_)];

                        if (perm_[0] != 0 || l != dim_ - 1) {
                            currentNode->next = new ComputeNode;
                            currentNode       = currentNode->next;
                        }
                    }

                    // macro-kernel
                    if (perm_[0] != 0) {
                        if (posStride1A_inB == 0)
                            currentNode->indexB = true;
                        currentNode->start     = -1;
                        currentNode->end       = -1;
                        currentNode->inc       = -1;
                        currentNode->lda       = lda_[posStride1B_inA];
                        currentNode->ldb       = ldb_[posStride1A_inB];
                        currentNode->offDiffAB = (ptrdiff_t)offsetA_[posStride1B_inA] - (ptrdiff_t)offsetB_[posStride1A_inB];
                        currentNode->next      = nullptr;
                    }
                }
                plans.push_back(plan);
                if (selectionMethod_ == ESTIMATE || selectionMethod_ == MEASURE && plans.size() > 200 ||
                    selectionMethod_ == PATIENT && plans.size() > 400 || selectionMethod_ == CRAZY && plans.size() > 800)
                    done = true;
            }
        }
}

/**
 * Estimates the time in seconds for the given computeTree
 */
template <typename floatType>
float Transpose<floatType>::estimateExecutionTime(std::shared_ptr<Plan> const plan) {
    auto startTime = std::chrono::high_resolution_clock::now();
    this->executeEstimate(plan.get());
    double elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
            .count();

    double const minMeasurementTime = 0.1; // in seconds

    // do at least 3 repetitions or spent at least 'minMeasurementTime' seconds
    // for each candidate
    int nRepeat = std::min(3, (int)std::ceil(minMeasurementTime / elapsedTime));

    // execute just a few iterations and exterpolate the result
    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nRepeat; ++i) // ATTENTION: we are not clearing the caches inbetween runs
        this->executeEstimate(plan.get());
    elapsedTime =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
            .count();
    elapsedTime /= nRepeat;

#ifdef DEBUG
    printf("Estimated time: %.3e ms.\n", elapsedTime);
#endif
    return elapsedTime;
}

template <typename floatType>
double Transpose<floatType>::getTimeLimit() const {
    if (selectionMethod_ == ESTIMATE)
        return 0.0;
    else if (selectionMethod_ == MEASURE)
        return 10.; // 10s
    else if (selectionMethod_ == PATIENT)
        return 60.; // 1m
    else if (selectionMethod_ == CRAZY)
        return 3600.; // 1h
    else {
        fprintf(stderr, "[HPTT] ERROR: selectionMethod unknown.\n");
        exit(-1);
    }
    return -1;
}

template <typename floatType>
std::shared_ptr<Plan> Transpose<floatType>::selectPlan(std::vector<std::shared_ptr<Plan>> const &plans) {
    if (plans.size() <= 0) {
        fprintf(stderr, "[HPTT] Internal error: not enough plans generated.\n");
        exit(-1);
    }
    if (selectionMethod_ == ESTIMATE) // fast return
        return plans[0];

    double timeLimit               = this->getTimeLimit() * 1000; // in ms
    int    maxAutotuningCandidates = plans.size();
    if (maxAutotuningCandidates_ != -1) {
        maxAutotuningCandidates = maxAutotuningCandidates_;
        timeLimit               = 1e9;
    }

    float minTime     = FLT_MAX;
    int   bestPlan_id = 0;

    if (plans.size() > 1) {
        int  plansEvaluated = 0;
        auto startTime      = std::chrono::high_resolution_clock::now();
        for (int plan_id = 0; plan_id < maxAutotuningCandidates; plan_id++) {
            auto p = plans[plan_id];

            double elapsedTime =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(std::chrono::high_resolution_clock::now() - startTime)
                    .count();
            if (elapsedTime >= timeLimit) // timelimit reached
                break;

            float estimatedTime = this->estimateExecutionTime(p);
            plansEvaluated++;

            if (estimatedTime < minTime) {
                bestPlan_id = plan_id;
                minTime     = estimatedTime;
            }
            if (this->infoLevel_ > 1) {
                printf("Plan %d will take roughly %f ms.\n", plan_id, estimatedTime * 1000.);
                plans[plan_id]->print();
            }
        }
        if (this->infoLevel_ > 0)
            printf("We evaluated %d/%lu candidates and selected candidate %d.\n", plansEvaluated, plans.size(), bestPlan_id);
    }
    return plans[bestPlan_id];
}

template <typename FloatType>
void Transpose<FloatType>::writeToFile(std::FILE *fp) const {
    setupFile(fp);

    // Get the file header.
    FileHeader header;
    uint32_t   check;
    size_t     error2;

    TransposeConstants constants;

    int error1 = fseek(fp, 0, SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    error2 = fread(&header, sizeof(FileHeader), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    if (strncmp(header.magic, "HPTT", 4) != 0) {
        throw std::runtime_error("Trying to write to a file that is not a HPTT transpose file!");
    }

    error1 = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    constants = {.dim                      = dim_,
                 .numThreads               = numThreads_,
                 .innerStrideA             = innerStrideA_,
                 .innerStrideB             = innerStrideB_,
                 .selectedParallelStrategy = selectedParallelStrategyId_,
                 .selectedLoopOrderId      = selectedLoopOrderId_,
                 .conjA                    = conjA_,
                 .pad                      = 0};

    error2 = fwrite(&constants, sizeof(TransposeConstants), 1, fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(sizeA_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(outerSizeA_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(outerSizeB_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(offsetA_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(offsetB_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(lda_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(ldb_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    error2 = fwrite(perm_.data(), sizeof(int), dim_, fp);

    if (error2 < dim_) {
        goto write_to_file_error;
    }

    std::fflush(fp);

    masterPlan_->writeToFile(fp);

    check = computeChecksum(fp);

    error1 = fseek(fp, offsetof(FileHeader, checksum), SEEK_SET);

    if (error1 != 0) {
        goto write_to_file_error;
    }

    error2 = fwrite(&check, sizeof(uint32_t), 1, fp);

    std::fflush(fp);

    if (error2 < 1) {
        goto write_to_file_error;
    }

    return;

write_to_file_error:
    perror("Error writing to file!");
    throw std::runtime_error("IO error");
}

template <typename FloatType>
Transpose<FloatType>::Transpose(std::FILE *fp, FloatType alpha, FloatType const *A, FloatType beta, FloatType *B) {
#ifdef _OPENMP
    omp_init_lock(&writelock);
#endif
    // Get the file header.
    FileHeader         header;
    size_t             error2;
    TransposeConstants constants;
    uint32_t           check;

    int error1 = fseek(fp, 0, SEEK_SET);

    if (error1 != 0) {
        goto read_from_file_error;
    }

    error2 = fread(&header, sizeof(FileHeader), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    if (strncmp(header.magic, "HPTT", 4) != 0) {
        throw std::runtime_error("Trying to read from a file that is not a HPTT transpose file!");
    }

    error1 = fseek(fp, sizeof(FileHeader), SEEK_SET);

    if (error1 != 0) {
        goto read_from_file_error;
    }

    error2 = fread(&constants, sizeof(TransposeConstants), 1, fp);

    if (error2 < 1) {
        goto read_from_file_error;
    }

    A_     = A;
    B_     = B;
    alpha_ = alpha;
    beta_  = beta;
    dim_   = constants.dim;

    if (endian_char() != header.version[3]) {
        dim_ = byteswap(dim_);
    }

    sizeA_.resize(dim_);
    perm_.resize(dim_);
    outerSizeA_.resize(dim_);
    outerSizeB_.resize(dim_);
    offsetA_.resize(dim_);
    offsetB_.resize(dim_);
    innerStrideA_ = constants.innerStrideA;
    innerStrideB_ = constants.innerStrideB;
    lda_.resize(dim_);
    ldb_.resize(dim_);
    threadIds_.reserve(dim_);
    numThreads_                 = constants.numThreads;
    selectedParallelStrategyId_ = constants.selectedParallelStrategy;
    selectedLoopOrderId_        = constants.selectedLoopOrderId;
    conjA_                      = constants.conjA;

    if (endian_char() != header.version[3]) {
        innerStrideA_               = byteswap(constants.innerStrideA);
        innerStrideB_               = byteswap(constants.innerStrideB);
        numThreads_                 = byteswap(constants.numThreads);
        selectedParallelStrategyId_ = byteswap(constants.selectedParallelStrategy);
        selectedLoopOrderId_        = byteswap(constants.selectedLoopOrderId);
    }

    for (int i = 0; i < numThreads_; ++i)
        threadIds_.push_back(i);

    error2 = fread(sizeA_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(outerSizeA_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(outerSizeB_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(offsetA_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(offsetB_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(lda_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(ldb_.data(), sizeof(size_t), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    error2 = fread(perm_.data(), sizeof(int), dim_, fp);

    if (error2 < dim_) {
        goto read_from_file_error;
    }

    if (endian_char() != header.version[3]) {
        for (int i = 0; i < dim_; i++) {
            sizeA_[i]      = byteswap(sizeA_[i]);
            perm_[i]       = byteswap(perm_[i]);
            outerSizeA_[i] = byteswap(outerSizeA_[i]);
            outerSizeB_[i] = byteswap(outerSizeB_[i]);
            offsetA_[i]    = byteswap(offsetA_[i]);
            offsetB_[i]    = byteswap(offsetB_[i]);
            lda_[i]        = byteswap(lda_[i]);
            ldb_[i]        = byteswap(ldb_[i]);
        }
    }

    masterPlan_ = std::make_shared<Plan>(fp, endian_char() != header.version[3]);

    return;

read_from_file_error:
    perror("Error reading from file!");
    throw std::runtime_error("IO error");
}

template class Transpose<float>;
template class Transpose<double>;
template class Transpose<FloatComplex>;
template class Transpose<DoubleComplex>;

template void Transpose<float>::execute_expert<true, true, true>();
template void Transpose<float>::execute_expert<true, false, true>();
template void Transpose<float>::execute_expert<false, true, true>();
template void Transpose<float>::execute_expert<false, false, true>();
template void Transpose<float>::execute_expert<true, true, false>();
template void Transpose<float>::execute_expert<true, false, false>();
template void Transpose<float>::execute_expert<false, true, false>();
template void Transpose<float>::execute_expert<false, false, false>();

template void Transpose<double>::execute_expert<true, true, true>();
template void Transpose<double>::execute_expert<false, true, true>();
template void Transpose<double>::execute_expert<true, false, true>();
template void Transpose<double>::execute_expert<false, false, true>();
template void Transpose<double>::execute_expert<true, true, false>();
template void Transpose<double>::execute_expert<false, true, false>();
template void Transpose<double>::execute_expert<true, false, false>();
template void Transpose<double>::execute_expert<false, false, false>();

template void Transpose<FloatComplex>::execute_expert<true, true, true>();
template void Transpose<FloatComplex>::execute_expert<false, true, true>();
template void Transpose<FloatComplex>::execute_expert<true, false, true>();
template void Transpose<FloatComplex>::execute_expert<false, false, true>();
template void Transpose<FloatComplex>::execute_expert<true, true, false>();
template void Transpose<FloatComplex>::execute_expert<false, true, false>();
template void Transpose<FloatComplex>::execute_expert<true, false, false>();
template void Transpose<FloatComplex>::execute_expert<false, false, false>();

template void Transpose<DoubleComplex>::execute_expert<true, true, true>();
template void Transpose<DoubleComplex>::execute_expert<false, true, true>();
template void Transpose<DoubleComplex>::execute_expert<true, false, true>();
template void Transpose<DoubleComplex>::execute_expert<false, false, true>();
template void Transpose<DoubleComplex>::execute_expert<true, true, false>();
template void Transpose<DoubleComplex>::execute_expert<false, true, false>();
template void Transpose<DoubleComplex>::execute_expert<true, false, false>();
template void Transpose<DoubleComplex>::execute_expert<false, false, false>();

} // namespace hptt
