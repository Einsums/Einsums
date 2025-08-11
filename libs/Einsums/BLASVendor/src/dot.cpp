//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Profile.hpp>

#include "Common.hpp"

namespace einsums::blas::vendor {

EINSUMS_DISABLE_WARNING_PUSH
EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
extern "C" {
extern float  FC_GLOBAL(sdot, SDOT)(int_t *, float const *, int_t *, float const *, int_t *);
extern double FC_GLOBAL(ddot, DDOT)(int_t *, double const *, int_t *, double const *, int_t *);
#ifdef EINSUMS_HAVE_MKL
extern void FC_GLOBAL(cdotc, CDOTC)(std::complex<float> *, int_t *, std::complex<float> const *, int_t *, std::complex<float> const *,
                                    int_t *);
extern void FC_GLOBAL(zdotc, ZDOTC)(std::complex<double> *, int_t *, std::complex<double> const *, int_t *, std::complex<double> const *,
                                    int_t *);
extern void FC_GLOBAL(cdotu, CDOTU)(std::complex<float> *, int_t *, std::complex<float> const *, int_t *, std::complex<float> const *,
                                    int_t *);
extern void FC_GLOBAL(zdotu, ZDOTU)(std::complex<double> *, int_t *, std::complex<double> const *, int_t *, std::complex<double> const *,
                                    int_t *);
#else
extern std::complex<float>  FC_GLOBAL(cdotc, CDOTC)(int_t *, std::complex<float> const *, int_t *, std::complex<float> const *, int_t *);
extern std::complex<double> FC_GLOBAL(zdotc, ZDOTC)(int_t *, std::complex<double> const *, int_t *, std::complex<double> const *, int_t *);
extern std::complex<float>  FC_GLOBAL(cdotu, CDOTU)(int_t *, std::complex<float> const *, int_t *, std::complex<float> const *, int_t *);
extern std::complex<double> FC_GLOBAL(zdotu, ZDOTU)(int_t *, std::complex<double> const *, int_t *, std::complex<double> const *, int_t *);
#endif
}
EINSUMS_DISABLE_WARNING_POP

auto sdot(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    LabeledSection(__func__);

    return FC_GLOBAL(sdot, SDOT)(&n, x, &incx, y, &incy);
}

auto ddot(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    LabeledSection(__func__);

    return FC_GLOBAL(ddot, DDOT)(&n, x, &incx, y, &incy);
}

// We implement the cdotu as the default for cdot.
auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    LabeledSection(__func__);

#ifdef EINSUMS_HAVE_MKL
    std::complex<float> out{0.0, 0.0};
    FC_GLOBAL(cdotu, CDOTU)(&out, &n, x, &incx, y, &incy);
    return out;
#else
    return FC_GLOBAL(cdotu, CDOTU)(&n, x, &incx, y, &incy);
#endif
}

// We implement the zdotu as the default for cdot.
auto zdot(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    LabeledSection(__func__);

#ifdef EINSUMS_HAVE_MKL
    std::complex<double> out{0.0, 0.0};
    FC_GLOBAL(zdotu, ZDOTU)(&out, &n, x, &incx, y, &incy);
    return out;
#else
    return FC_GLOBAL(zdotu, ZDOTU)(&n, x, &incx, y, &incy);
#endif
}

auto cdotc(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    LabeledSection(__func__);

#ifdef EINSUMS_HAVE_MKL
    std::complex<float> out{0.0, 0.0};
    FC_GLOBAL(cdotc, CDOTC)(&out, &n, x, &incx, y, &incy);
    return out;
#else
    return FC_GLOBAL(cdotc, CDOTC)(&n, x, &incx, y, &incy);
#endif
}

auto zdotc(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy) -> std::complex<double> {
    LabeledSection(__func__);

#ifdef EINSUMS_HAVE_MKL
    std::complex<double> out{0.0, 0.0};
    FC_GLOBAL(zdotc, ZDOTC)(&out, &n, x, &incx, y, &incy);
    return out;
#else
    return FC_GLOBAL(zdotc, ZDOTC)(&n, x, &incx, y, &incy);
#endif
}

} // namespace einsums::blas::vendor