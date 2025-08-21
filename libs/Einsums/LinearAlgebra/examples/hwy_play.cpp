//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "hwy_play.cpp"
#include <hwy/foreach_target.h> // IWYU pragma: keep
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace einsums {
namespace HWY_NAMESPACE {
namespace hn = ::hwy::HWY_NAMESPACE;

// A, B, and C cannot overlap and padded to multiples of lane size.
template <typename T>
void hadamard_product_kernel(T alpha, T const *HWY_RESTRICT A, T const *HWY_RESTRICT B, T *HWY_RESTRICT C, size_t N) {

    // Runtime check to ensure N is a multiple of Lanes

    using RT = RemoveComplexT<T>;
    using D  = hn::ScalableTag<RemoveComplexT<RT>>;

    D const      d;
    size_t const L = hn::Lanes(d);

    // This printing can be removed.
    // println("---------------------- {} - Lanes {}", hwy::TargetName(HWY_TARGET), hn::Lanes(d));

    // if (N % L != 0) {
    //    EINSUMS_THROW_EXCEPTION(std::runtime_error, "einsums::hadamard_product_kernel: N is not a multiple of SIMD vector length");
    // }

    if constexpr (!IsComplexV<T>) {
        // ------ Real path (float/double)
        auto const valpha = hn::Set(d, alpha);

        size_t i = 0;

        // Main vector loop
        for (; i <= N; i += L) {
            // Once the tensor is guaranteed to be aligned on the lane boundary this can be changed to Load and Store
            auto a = hn::Load(d, A + i);
            auto b = hn::Load(d, B + i);
            auto c = hn::Load(d, C + i);
            c      = hn::MulAdd(hn::Mul(valpha, a), b, c); // c += (alpha*a)*b
            hn::Store(c, d, C + i);
        }
    } else {
        // ------ Complex path (std::complex<float/double>)
        auto const alpha_re = hn::Set(d, alpha.real());
        auto const alpha_im = hn::Set(d, alpha.imag());

        size_t i = 0;

        for (; i < N; i += L) {
            auto a_re = hn::Zero(d), a_im = hn::Zero(d);
            auto b_re = hn::Zero(d), b_im = hn::Zero(d);
            auto c_re = hn::Zero(d), c_im = hn::Zero(d);

            hn::LoadInterleaved2(d, reinterpret_cast<const RT *>(A) + 2 * i, a_re, a_im); // aligned
            hn::LoadInterleaved2(d, reinterpret_cast<const RT *>(B) + 2 * i, b_re, b_im);
            hn::LoadInterleaved2(d, reinterpret_cast<const RT *>(C) + 2 * i, c_re, c_im);

            // alpha * a
            auto tmp_re = hn::Sub(hn::Mul(alpha_re, a_re), hn::Mul(alpha_im, a_im));
            auto tmp_im = hn::Add(hn::Mul(alpha_re, a_im), hn::Mul(alpha_im, a_re));

            // c += (alpha * a) * b
            auto prod_re = hn::Sub(hn::Mul(tmp_re, b_re), hn::Mul(tmp_im, b_im));
            auto prod_im = hn::Add(hn::Mul(tmp_re, b_im), hn::Mul(tmp_im, b_re));

            c_re = hn::Add(c_re, prod_re);
            c_im = hn::Add(c_im, prod_im);

            hn::StoreInterleaved2(c_re, c_im, d, reinterpret_cast<RT *>(C) + 2 * i); // aligned store
        }
    }
}

void hadamard_product_double(double alpha, double const *A, double const *B, double *C, size_t N) {
    hadamard_product_kernel(alpha, A, B, C, N);
}

// transpose kernel
template <bool UseIdxA, bool UseIdxB, bool betaIsZero, bool conjA, typename T>
struct TransposeMicroKernel {
    void operator()(T const *A, size_t lda, int innerStrideA, T *B, size_t ldb, int innerStrideB, T alpha, T beta) const {
        using namespace hn;

        using D = ScalableTag<RemoveComplexT<T>>;
        D const                            d;
        size_t const                       L = Lanes(d);
        Rebind<int32_t, decltype(d)> const di;

        auto const                  v_alpha = Set(d, alpha);
        [[maybe_unused]] auto const v_beta  = Set(d, beta);

        // Build index vectors once if we need strided access
        [[maybe_unused]] auto const idxA = Mul(Set(di, innerStrideA), Iota(di, 0));
        [[maybe_unused]] auto const idxB = Mul(Set(di, innerStrideB), Iota(di, 0));

        // Helpers: load/store with/without indices (no branches in the loop)
        auto loadA = [&](T const *base) {
            if constexpr (UseIdxA)
                return Gather(d, base, idxA);
            else
                return Load(d, base);
        };
        auto loadB = [&](T const *base) {
            if constexpr (UseIdxB)
                return Gather(d, base, idxB);
            else
                Store(d, base);
        };
        auto storeB = [&](auto v, T *base) {
            if constexpr (UseIdxB)
                Scatter(v, d, base, idxB);
            else
                Store(v, d, base);
        };

        // Process a 4x4 block: load 4 rows of A
        auto a0 = loadA(A + 0 * lda);
        auto a1 = loadA(A + 1 * lda);
        auto a2 = loadA(A + 2 * lda);
        auto a3 = loadA(A + 3 * lda);

        if constexpr (conjA) {
            // no-op for real, placeholder for complex later.
        }

        // Transpose 4x4 of values: (a0...a3) -> (t0..t3)
        auto const ab_lo = InterleaveLower(a0, a1); // [a0, b0, a1, b1]
        auto const ab_hi = InterleaveUpper(a0, a1); // [a2, b2, a3, b3]
        auto const cd_lo = InterleaveLower(a2, a3); // [c0, d0, c1, d1]
        auto const cd_hi = InterleaveUpper(a2, a3); // [c2, d2, c3, d3]

        auto const t0 = Combine(LowerHalf(cd_lo), LowerHalf(ab_lo)); // [a0, b0, c0, d0]
        auto const t1 = Combine(UpperHalf(cd_lo), UpperHalf(ab_lo)); // [a1, b1, c1, d1]
        auto const t2 = Combine(LowerHalf(cd_hi), UpperHalf(ab_hi)); // [a2, b2, c2, d2]
        auto const t3 = Combine(UpperHalf(cd_hi), UpperHalf(ab_hi)); // [a3, b3, c3, d3]

        // Scale by alpha
        auto r0 = t0 * v_alpha;
        auto r1 = t1 * v_alpha;
        auto r2 = t2 * v_alpha;
        auto r3 = t3 * v_alpha;

        if constexpr (!betaIsZero) {
            // B = beta*B + r
            auto b0 = loadB(B + 0 * ldb);
            auto b1 = loadB(B + 1 * ldb);
            auto b2 = loadB(B + 2 * ldb);
            auto b3 = loadB(B + 3 * ldb);

            b0 = b0 * v_beta + r0;
            b1 = b1 * v_beta + r1;
            b2 = b2 * v_beta + r2;
            b3 = b3 * v_beta + r3;

            storeB(b0, B + 0 * ldb);
            storeB(b1, B + 1 * ldb);
            storeB(b2, B + 2 * ldb);
            storeB(b3, B + 3 * ldb);

        } else {
            // B = r
            storeB(r0, B + 0 * ldb);
            storeB(r1, B + 1 * ldb);
            storeB(r2, B + 2 * ldb);
            storeB(r3, B + 3 * ldb);
        }
    }
};

} // namespace HWY_NAMESPACE
} // namespace einsums
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace einsums {
namespace {
HWY_EXPORT(hadamard_product_double);

void hadamard_product(double alpha, Tensor<double, 1> const &A, Tensor<double, 1> const &B, Tensor<double, 1> *C) {
    HWY_DYNAMIC_DISPATCH(hadamard_product_double)(alpha, A.data(), B.data(), C->data(), A.size());
}
} // namespace
int einsums_main() {
    using namespace einsums;

    size_t i{8};

    auto A = create_incremented_tensor("A", i);
    auto B = create_incremented_tensor("B", i);
    auto C = create_zero_tensor("C", i);

    hadamard_product(1.0, A, B, &C);

    println(A);
    println(B);
    println(C);

    finalize();
    return EXIT_SUCCESS;
} // namespace int einsums_main()
} // namespace einsums

int main(int argc, char **argv) {
    return einsums::start(einsums::einsums_main, argc, argv);
}
#endif