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
void dirprod_kernel(T alpha, T const *HWY_RESTRICT A, T const *HWY_RESTRICT B, T *HWY_RESTRICT C, size_t N) {

    // Runtime check to ensure N is a multiple of Lanes

    using RT = RemoveComplexT<T>;
    using D  = hn::ScalableTag<RemoveComplexT<RT>>;

    D const      d;
    size_t const L = hn::Lanes(d);

    // This printing can be removed.
    // println("---------------------- {} - Lanes {}", hwy::TargetName(HWY_TARGET), hn::Lanes(d));

    // if (N % L != 0) {
    //    EINSUMS_THROW_EXCEPTION(std::runtime_error, "einsums::dirprod_kernel: N is not a multiple of SIMD vector length");
    // }

    if constexpr (!IsComplexV<T>) {
        // ------ Real path (float/double)
        auto const valpha = hn::Set(d, alpha);

        size_t i = 0;

        // Main vector loop
        for (; i <= N; i += L) {
            // Once the tensor is guaranteed to be aligned on the lane boundary this can be changed to Load and Store
            auto a = hn::LoadU(d, A + i);
            auto b = hn::LoadU(d, B + i);
            auto c = hn::LoadU(d, C + i);
            c      = hn::MulAdd(hn::Mul(valpha, a), b, c); // c += (alpha*a)*b
            hn::StoreU(c, d, C + i);
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

void dirprod_double(double alpha, double const *A, double *B, double *C, size_t N) {
    dirprod_kernel(alpha, A, B, C, N);
}

} // namespace HWY_NAMESPACE
} // namespace einsums
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace einsums {
namespace {
HWY_EXPORT(dirprod_double);
}
int einsums_main() {
    using namespace einsums;

    size_t i{8};

    auto A = create_random_tensor("A", i);
    auto B = create_random_tensor("B", i);
    auto C = create_zero_tensor("C", i);

    HWY_DYNAMIC_DISPATCH(dirprod_double)(1.0, A.data(), B.data(), C.data(), A.size());

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