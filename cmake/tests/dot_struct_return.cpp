#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using namespace std;

#if defined(EINSUMS_BLAS_INTERFACE_ILP64)
using int_t   = long long int;
using euint_t = unsigned long long int;
using elong   = long long int;
#elif defined(EINSUMS_BLAS_INTERFACE_LP64)
using int_t   = int;
using euint_t = unsigned int;
using elong   = long int;
#else
using int_t   = int;
using euint_t = unsigned int;
using elong   = long int;
#endif

#ifndef FC_SYMBOL
#    define FC_SYMBOL 2
#endif

#if FC_SYMBOL == 1
#    define FC_GLOBAL(name, NAME) name
#elif FC_SYMBOL == 2
#    define FC_GLOBAL(name, NAME) name##_
#elif FC_SYMBOL == 3
#    define FC_GLOBAL(name, NAME) NAME
#elif FC_SYMBOL == 4
#    define FC_GLOBAL(name, NAME) NAME##_
#endif

struct complex_float_t {
    float r, i;
};

extern "C" {
extern complex_float_t FC_GLOBAL(cdotu, CDOTU)(int_t *, std::complex<float> const *, int_t *, std::complex<float> const *, int_t *);
}

auto cdot(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy) -> std::complex<float> {
    struct {
        int_t                      n;
        std::complex<float> const *x;
        int_t                      incx;
        std::complex<float> const *y;
        int_t                      incy;
    } args{n, x, incx, y, incy};

    std::complex<float> test{0.0};

    for (int i = 0; i < n; i++) {
        test += x[i * incx] * y[i * incy];
    }

    complex_float_t out = FC_GLOBAL(cdotu, CDOTU)(&args.n, args.x, &args.incx, args.y, &args.incy);
    std::complex<float> result{out.r, out.i};

    if (std::abs(test - result) > 1e-4) {
        printf("Error 1");
        std::exit(-1);
    }

    if (args.n != n || args.x != x || args.incx != incx || args.y != y || args.incy != incy) {
        printf("Error 2");
        std::exit(-2);
    }

    printf("No error");

    return result;
}

int main(void) {
    std::complex<float> arr1[10], arr2[10];

    for (int i = 0; i < 10; i++) {
        arr1[i] = std::complex<float>{(float)i};
        arr2[i] = std::complex<float>{(float)10 * i};
    }

    auto ret = cdot(10, arr1, 1, arr2, 1);

    return 0;
}
