
#define EINSUMS_PRAGMA(stuff) _Pragma(#stuff)
#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    define EINSUMS_OMP_PRAGMA(stuff)      EINSUMS_PRAGMA(omp stuff)
#    define EINSUMS_OMP_SIMD_PRAGMA(stuff) EINSUMS_PRAGMA(omp stuff simd)
#    define EINSUMS_OMP_SIMD               _Pragma("omp simd")
#elif 0
#    define EINSUMS_OMP_PRAGMA(stuff)      EINSUMS_PRAGMA(omp stuff)
#    define EINSUMS_OMP_SIMD_PRAGMA(stuff) EINSUMS_PRAGMA(omp stuff)
#    define EINSUMS_OMP_SIMD
#endif

#if 0
#    define EINSUMS_OMP_PARALLEL_FOR_SIMD EINSUMS_OMP_SIMD_PRAGMA(parallel for)
#    define EINSUMS_OMP_PARALLEL_FOR EINSUMS_OMP_PRAGMA(parallel for)
#    define EINSUMS_OMP_PARALLEL      EINSUMS_OMP_PRAGMA(parallel)
#    define EINSUMS_OMP_TASK_FOR      EINSUMS_OMP_PRAGMA(taskloop)
#    define EINSUMS_OMP_TASK_FOR_SIMD EINSUMS_OMP_SIMD_PRAGMA(taskloop)
#    define EINSUMS_OMP_TASK          EINSUMS_OMP_PRAGMA(task)
#    define EINSUMS_OMP_FOR_NOWAIT EINSUMS_OMP_PRAGMA(for nowait)
#    define EINSUMS_OMP_CRITICAL EINSUMS_OMP_PRAGMA(critical)
#else
#    define EINSUMS_OMP_PRAGMA(stuff)
#    define EINSUMS_OMP_SIMD_PRAGMA(stuff)
#    define EINSUMS_OMP_SIMD
#    define EINSUMS_OMP_PARALLEL_FOR_SIMD
#    define EINSUMS_OMP_PARALLEL_FOR
#    define EINSUMS_OMP_PARALLEL
#    define EINSUMS_OMP_TASK_FOR
#    define EINSUMS_OMP_TASK_FOR_SIMD
#    define EINSUMS_OMP_TASK
#    define EINSUMS_OMP_FOR_NOWAIT
#    define EINSUMS_OMP_CRITICAL
#endif

#include <stddef.h>

#if !defined(__AVX2__)
void sdirprod(size_t n, float alpha, float const *__restrict__ x, size_t incx, float const *__restrict__ y, size_t incy, float beta,
              float *__restrict__ z, size_t incz) {
    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < n; i++) {
        z[i * incz] = alpha * z[i * incz] + beta * x[i * incx] * y[i * incy];
    }
}

#endif