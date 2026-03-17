#include <complex>

#if defined(EINSUMS_BLAS_INTERFACE_ILP64)
using int_t   = long long int;
using euint_t = unsigned long long int;
using elong   = long long int;
#elif defined(EINSUMS_BLAS_INTERFACE_LP64)
using int_t   = int;
using euint_t = unsigned int;
using elong   = long int;
#else
using int_t = int;
using euint_t = unsigned int;
using elong = long int;
#endif

#ifndef FC_SYMBOL
#define FC_SYMBOL 2
#endif

#if FC_SYMBOL == 1
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) name
#elif FC_SYMBOL == 2
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) name##_
#elif FC_SYMBOL == 3
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) NAME
#elif FC_SYMBOL == 4
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) NAME##_
#endif

extern void FC_GLOBAL(cdotu, CDOTU)(std::complex<float>*, int_t*, std::complex<float> const*, int_t*, std::complex<float> const*, int_t*);

auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    std::complex<float> out { 0.0, 0.0 };
    FC_GLOBAL(cdotu, CDOTU)(&out, &n, x, &incx, y, &incy);
    return out;
}

int main(void) {
    std::complex<float> arr1[10], arr2[10];

    auto ret = cdot(10, arr1, 1, arr2, 1);

    return 0;
}
