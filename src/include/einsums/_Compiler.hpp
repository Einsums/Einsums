#pragma once

#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
#    define EINSUMS_OMP_SIMD         _Pragma("omp simd")
#    define EINSUMS_OMP_PARALLEL     _Pragma("omp parallel")
#else
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#    define EINSUMS_OMP_SIMD
#    define EINSUMS_OMP_PARALLEL _Pragma("omp parallel")
#endif

#define EINSUMS_ALWAYS_INLINE __attribute__((always_inline)) inline