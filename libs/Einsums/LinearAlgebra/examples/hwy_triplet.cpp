//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

#include <queue>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy_triplet.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace einsums {
namespace HWY_NAMESPACE {
namespace hn = ::hwy::HWY_NAMESPACE;

// -----------------
// Tuning Parameters
// -----------------
static constexpr size_t TILE_I   = 8;  // rows of A / D processed per i-block
static constexpr size_t TILE_J   = 64; // j-block (iterates over N)
static constexpr size_t TILE_K   = 64; // k-block (iterates over P)
static constexpr size_t UNROLL_L = 2;  // how many SIMD vectors per l-block
// -----------------

static void *aligned_alloc64(size_t bytes) {
    void *p = nullptr;
#if defined(_ISOC11_SOURCE) || (__STDC_VERSION__ >= 201112L)
    p = std::aligned_alloc(64, bytes);
    if (!p) {
        std::cerr << "aligned_alloc failed\n";
        std::exit(1);
    }
#else
    if (posix_memalign(&p, 64, bytes) != 0) {
        std::cerr << "posix_memalign failed\n";
        std::exit(1);
    }
#endif
    return p;
}

// ----------------- Packing / Unpacking Helpers
template <bool TA>
static inline void pack_A(double const *A, size_t M, size_t N, size_t i0, size_t j0, size_t ib, size_t jb, double *Ap) {
    // Pack Ap as (ib x TILE_J), row stride TILE_J; zero outside (ib x jb)
    for (size_t ii = 0; ii < ib; ++ii) {
        double *dst = Ap + ii * TILE_J;
        // copy valid jb entries
        for (size_t j = 0; j < jb; ++j) {
            // logical a(i0+ii, j0+j)
            size_t si, sj;
            if constexpr (!TA) {
                si = i0 + ii;
                sj = j0 + j; /* A[MxN] */
            } else {
                si = j0 + j;
                sj = i0 + ii; /* A^T [NxM] */
            }
            size_t const stride_cols = (TA ? M : N);
            dst[j]                   = A[si * stride_cols + sj];
        }
        // zero-pad remainder of row
        if (jb < TILE_J)
            std::memset(dst + jb, 0, (TILE_J - jb) * sizeof(double));
    }
    // zero-pad remaining rows
    for (size_t ii = ib; ii < TILE_I; ++ii)
        std::memset(Ap + ii * TILE_J, 0, TILE_J * sizeof(double));
}

template <bool TB>
static inline void pack_B(double const *B, size_t N, size_t K, size_t j0, size_t k0, size_t jb, size_t kb, double *Bp) {
    // Pack Bp as (jb x TILE_K), row stride TILE_K; zero outside (jb x kb).
    for (size_t j = 0; j < jb; ++j) {
        double *dst = Bp + j * TILE_K;
        for (size_t k = 0; k < kb; ++k) {
            size_t si, sj;
            if constexpr (!TB) {
                si = j0 + j;
                sj = k0 + k; /* B[NxK] */
            } else {
                si = k0 + k;
                sj = j0 + j; /* B^T[KxN] */
            }
            size_t const stride_cols = (TB ? N : K);
            dst[k]                   = B[si * stride_cols + sj];
        }
        if (kb < TILE_K)
            std::memset(dst + kb, 0, (TILE_K - kb) * sizeof(double));
    }
    for (size_t j = jb; j < TILE_J; ++j)
        std::memset(Bp + j * TILE_K, 0, TILE_K * sizeof(double));
}

template <bool TC>
static inline void pack_C(double const *C, size_t K, size_t L, size_t k0, size_t l0, size_t kb, size_t Lb, double *Cp) {
    // Pack Cp as (kb x Lb), row stride Lb; zero outside (kb x Lb_eff).
    for (size_t k = 0; k < kb; ++k) {
        double *dst = Cp + k * Lb;
        for (size_t t = 0; t < Lb; ++t) {
            size_t si, sj;
            if constexpr (!TC) {
                si = k0 + k;
                sj = l0 + t; /* C[KxL] */
            } else {
                si = l0 + t;
                sj = k0 + k; /* C^T[LxK] */
            }
            size_t const stride_cols = (TC ? K : L);
            // If l0+t >= L (panel tail), we read OOB logically; write zero instead.
            if ((!TC && (l0 + t) >= L) || (TC && (l0 + t) >= L)) {
                dst[t] = 0.0;
            } else {
                dst[t] = C[si * stride_cols + sj];
            }
        }
    }
    for (size_t k = kb; k < TILE_K; ++k)
        std::memset(Cp + k * Lb, 0, Lb * sizeof(double));
}

static inline void pack_D(double const *D, size_t M, size_t L, size_t i0, size_t l0, size_t ib, size_t Lb, double *Dp) {
    // Pack Dp as (ib x Lb), row stride Lb; zero beyond valid L.
    for (size_t ii = 0; ii < ib; ++ii) {
        double      *dst     = Dp + ii * Lb;
        size_t const l_valid = std::min(Lb, (l0 < L) ? (L - l0) : size_t(0));
        if (l_valid)
            std::memcpy(dst, D + (i0 + ii) * L + l0, l_valid * sizeof(double));
        if (l_valid < Lb)
            std::memset(dst + l_valid, 0, (Lb - l_valid) * sizeof(double));
    }
    for (size_t ii = ib; ii < TILE_I; ++ii)
        std::memset(Dp + ii * Lb, 0, Lb * sizeof(double));
}

static inline void unpack_D(double const *Dp, size_t M, size_t L, size_t i0, size_t l0, size_t ib, size_t Lb, double *D) {
    size_t const l_valid = std::min(Lb, (l0 < L) ? (L - l0) : size_t(0));
    for (size_t ii = 0; ii < ib; ++ii) {
        if (l_valid)
            std::memcpy(D + (i0 + ii) * L + l0, Dp + ii * Lb, l_valid * sizeof(double));
    }
}

static inline void fused_kernel_packed(double const *Ap, double const *Bp, double const *Cp, double *Dp, size_t ib, size_t jb, size_t kb,
                                       size_t Lb) {
    using namespace hn;

    ScalableTag<double> const d;
    size_t const              lanes = Lanes(d);
    (void)lanes;
    alignas(64) Vec<decltype(d)> acc[TILE_I][UNROLL_L];

    // load Dp accumulators
    for (size_t ii = 0; ii < TILE_I; ++ii)
        for (size_t u = 0; u < UNROLL_L; ++u)
            acc[ii][u] = Load(d, Dp + ii * Lb + u * lanes);

    for (size_t k = 0; k < kb; ++k) {
        Vec<decltype(d)> cvec[UNROLL_L];
        for (size_t u = 0; u < UNROLL_L; ++u)
            cvec[u] = Load(d, Cp + k * Lb + u * lanes);

        for (size_t j = 0; j < jb; ++j) {
            auto const       vb = Set(d, Bp[j * TILE_K + k]);
            Vec<decltype(d)> bc[UNROLL_L];
            for (size_t u = 0; u < UNROLL_L; ++u)
                bc[u] = Mul(cvec[u], vb);

            for (size_t ii = 0; ii < ib; ++ii) {
                auto const va = Set(d, Ap[ii * TILE_J + j]);
                for (size_t u = 0; u < UNROLL_L; ++u)
                    acc[ii][u] = MulAdd(va, bc[u], acc[ii][u]);
            }
        }
    }

    for (size_t ii = 0; ii < TILE_I; ++ii)
        for (size_t u = 0; u < UNROLL_L; ++u)
            Store(acc[ii][u], d, Dp + ii * Lb + u * lanes);
}

// ---------------- Template specialization: 8 combinations (TA, TB, TC)
template <bool TA, bool TB, bool TC>
void triple_matmul_fused_highway_T(double const *A, double const *B, double const *C, double *D, size_t M, size_t N, size_t K, size_t L) {
    using namespace hn;

    ScalableTag<double> const d;
    size_t const              lanes = Lanes(d);
    size_t const              Lb    = lanes * UNROLL_L;

    // Scratch panels
    double *Ap = static_cast<double *>(aligned_alloc64(TILE_I * TILE_J * sizeof(double)));
    double *Bp = static_cast<double *>(aligned_alloc64(TILE_J * TILE_K * sizeof(double)));
    double *Cp = static_cast<double *>(aligned_alloc64(TILE_K * Lb * sizeof(double)));
    double *Dp = static_cast<double *>(aligned_alloc64(TILE_I * Lb * sizeof(double)));

    for (size_t i0 = 0; i0 < M; i0 += TILE_I) {
        size_t const ib = std::min(TILE_I, M - i0);
        for (size_t l0 = 0; l0 < L; l0 += Lb) {
            pack_D(D, M, L, i0, l0, ib, Lb, Dp);

            for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                size_t const kb = std::min(TILE_K, K - k0);
                pack_C<TC>(C, K, L, k0, l0, kb, Lb, Cp);

                for (size_t j0 = 0; j0 < N; j0 += TILE_J) {
                    size_t const jb = std::min(TILE_J, N - j0);
                    pack_A<TA>(A, M, N, i0, j0, ib, jb, Ap);
                    pack_B<TB>(B, N, K, j0, k0, jb, kb, Bp);

                    fused_kernel_packed(Ap, Bp, Cp, Dp, ib, jb, kb, Lb);
                }
            }

            unpack_D(Dp, M, L, i0, l0, ib, Lb, D);
        }
    }

    free(Ap);
    free(Bp);
    free(Cp);
    free(Dp);
}

void triple_matmul_fused_highway_dispatch(double const *A, double const *B, double const *C, double *D, size_t M, size_t N, size_t K,
                                          size_t L, bool TA, bool TB, bool TC) {
    // Choose specialization once (no inner branching)
    if (!TA && !TB && !TC)
        return triple_matmul_fused_highway_T<false, false, false>(A, B, C, D, M, N, K, L);
    if (!TA && !TB && TC)
        return triple_matmul_fused_highway_T<false, false, true>(A, B, C, D, M, N, K, L);
    if (!TA && TB && !TC)
        return triple_matmul_fused_highway_T<false, true, false>(A, B, C, D, M, N, K, L);
    if (!TA && TB && TC)
        return triple_matmul_fused_highway_T<false, true, true>(A, B, C, D, M, N, K, L);
    if (TA && !TB && !TC)
        return triple_matmul_fused_highway_T<true, false, false>(A, B, C, D, M, N, K, L);
    if (TA && !TB && TC)
        return triple_matmul_fused_highway_T<true, false, true>(A, B, C, D, M, N, K, L);
    if (TA && TB && !TC)
        return triple_matmul_fused_highway_T<true, true, false>(A, B, C, D, M, N, K, L);
    /* TA && TB && TC */ return triple_matmul_fused_highway_T<true, true, true>(A, B, C, D, M, N, K, L);
}

} // namespace HWY_NAMESPACE
} // namespace einsums
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace einsums {

struct ThreadPool {
    ThreadPool(size_t n) : _stop(false) {
        for (size_t i = 0; i < n; ++i) {
            _workers.emplace_back([this] { this->worker_loop(); });
        }
    }
    ~ThreadPool() {
        {
            std::unique_lock lk(_mu);
            _stop = true;
            _cv.notify_all();
        }
        for (auto &w : _workers)
            if (w.joinable())
                w.join();
    }

  private:
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lk(_mu);
                _cv.wait(lk, [&] { return _stop || !_tasks.empty(); });
                if (_stop && _tasks.empty())
                    return;
                task = std::move(_tasks.front());
                _tasks.pop();
            }
            task();
        }
    }

    std::vector<std::thread>          _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex                        _mu;
    std::condition_variable           _cv;
    bool                              _stop;
};

namespace {
HWY_EXPORT(triple_matmul_fused_highway_dispatch);

void triplet(bool TA, Tensor<double, 2> const &A, bool TB, Tensor<double, 2> const &B, bool TC, Tensor<double, 2> C, Tensor<double, 2> *D) {
    LabeledSection("Triplet<{},{},{}>", TA, TB, TC);
    auto i = A.dim(0);
    HWY_DYNAMIC_DISPATCH(triple_matmul_fused_highway_dispatch)(A.data(), B.data(), C.data(), D->data(), i, i, i, i, TA, TB, TC);
}

// -------------------------------
// Reference scalar implementation
// A(M, N), B(N, P), C(P, Q) -> D(M, Q)
// ------------------------)-------
void triplet_ref(bool TA, Tensor<double, 2> const &A, bool TB, Tensor<double, 2> const &B, bool TC, Tensor<double, 2> &C,
                 Tensor<double, 2> *D) {
    LabeledSection("Triplet Reference<{},{},{}>", TA, TB, TC);
    Tensor<double, 2> &Dref = *D;

    size_t M = A.dim(0), N = A.dim(1);
    size_t P = B.dim(1);
    size_t Q = C.dim(1);
    for (size_t i = 0; i < M; ++i) {
        for (size_t l = 0; l < Q; ++l) {
            double sum = 0.0;
            for (size_t j = 0; j < N; ++j) {
                for (size_t k = 0; k < P; ++k) {
                    double const a = TA ? A(j, i) : A(i, j);
                    double const b = TB ? B(k, j) : B(j, k);
                    double const c = TC ? C(l, k) : C(k, l);
                    sum += a * b * c;
                    ;
                }
            }
            Dref(i, l) = sum;
        }
    }
}

bool almost_equal(double const *a, double const *b, size_t n, double eps = 1e-9) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            std::cerr << "Mismatch @" << i << " a=" << a[i] << " b=" << b[i] << "\n";
            return false;
        }
    }
    return true;
}

} // namespace

int einsums_main() {
    using namespace einsums;

    // Currently dimensions need to be at least 64
    size_t i{256};

    auto A    = create_random_tensor("A", i, i);
    auto B    = create_random_tensor("B", i, i);
    auto C    = create_random_tensor("C", i, i);
    auto D    = create_zero_tensor("D", i, i);
    auto Dref = create_zero_tensor("Dref", i, i);

    bool const flags[2] = {false, true};
    for (bool TA : flags) {
        for (bool TB : flags) {
            for (bool TC : flags) {
                D.zero();
                Dref.zero();

                triplet(TA, A, TB, B, TC, C, &D);
                triplet_ref(TA, A, TB, B, TC, C, &Dref);
                if (almost_equal(D.data(), Dref.data(), D.size())) {
                    println("Looks good!!!");
                }
            }
        }
    }

    // println(Dref);
    // println(D);

    finalize();
    return EXIT_SUCCESS;
}
} // namespace einsums

int main(int argc, char **argv) {
    return einsums::start(einsums::einsums_main, argc, argv);
}
#endif
