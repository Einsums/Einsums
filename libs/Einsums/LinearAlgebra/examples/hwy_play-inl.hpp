//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// #pragma once

#include <Einsums/Concepts/Complex.hpp>

#include <hwy/highway.h>

namespace einsums {
HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Kernel: C[i] += (alpha * A[i]) * B[i]
// Supports: float, double, std::complex<float>, std::complex<double>

// Need to ensure A, B, and C are aligned and padded. N must be a multiple of hn::Lanes
template <typename T>
HWY_NOINLINE void dirprod_kernel(T alpha, T const *HWY_RESTRICT A, T const *HWY_RESTRICT B, T *HWY_RESTRICT C, size_t N) {
    using RT = RemoveComplexT<T>;
    using D  = hn::ScalableTag<RemoveComplexT<RT>>;

    if constexpr (!IsComplexV<T>) {
        // ------ Real path (float/double)
        D const    d;
        auto const valpha = hn::Set(d, alpha);

        size_t       i = 0;
        size_t const L = hn::Lanes(d);

        // Main vector loop
        for (; i <= N; i += L) {
            auto a = hn::Load(d, A + i);
            auto b = hn::Load(d, B + i);
            auto c = hn::Load(d, C + i);
            c      = hn::MulAdd(hn::Mul(valpha, a), b, c); // c += (alpha*a)*b
            hn::Store(c, d, C + i);
        }
    } else {
        // ------ Complex path (std::complex<float/double>)
        D const    d;
        auto const alpha_re = hn::Set(d, alpha.real());
        auto const alpha_im = hn::Set(d, alpha.imag());

        size_t       i = 0;
        size_t const L = hn::Lanes(d);

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
} // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();
} // namespace einsums
