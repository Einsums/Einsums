//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <complex>

#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
#    define EINSUMS_OMP_SIMD         _Pragma("omp simd")
#    define EINSUMS_OMP_PARALLEL     _Pragma("omp parallel")
#    define EINSUMS_OMP_TASK_FOR     _Pragma("omp taskloop simd")
#    define EINSUMS_OMP_TASK         _Pragma("omp task")
#    define EINSUMS_OMP_FOR_NOWAIT   _Pragma("omp for nowait")
#    define EINSUMS_OMP_CRITICAL     _Pragma("omp critical")
#    define EINSUMS_SIMD_ENABLED
#else
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#    define EINSUMS_OMP_SIMD
#    define EINSUMS_OMP_PARALLEL   _Pragma("omp parallel")
#    define EINSUMS_OMP_TASK_FOR   _Pragma("omp taskloop")
#    define EINSUMS_OMP_TASK       _Pragma("omp task")
#    define EINSUMS_OMP_FOR_NOWAIT _Pragma("omp for nowait")
#    define EINSUMS_OMP_CRITICAL   _Pragma("omp critical")
#endif

#ifdef __GNUC__

// gcc does not have reductions for complex values.
#    pragma omp declare reduction(+ : std::complex<float> : omp_out += omp_in) initializer(omp_priv = omp_orig)
#    pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in) initializer(omp_priv = omp_orig)

#endif

#define EINSUMS_ALWAYS_INLINE __attribute__((always_inline)) inline

// clang-format off
#if defined(_MSC_VER)
#    define EINSUMS_DISABLE_WARNING_PUSH           __pragma(warning(push))
#    define EINSUMS_DISABLE_WARNING_POP            __pragma(warning(pop))
#    define EINSUMS_DISABLE_WARNING(warningNumber) __pragma(warning(disable : warningNumber))

#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
// other warnings you want to deactivate...

#elif defined(__GNUC__) || defined(__clang__)
#    define EINSUMS_DO_PRAGMA(X)                 _Pragma(#X)
#    define EINSUMS_DISABLE_WARNING_PUSH         EINSUMS_DO_PRAGMA(GCC diagnostic push)
#    define EINSUMS_DISABLE_WARNING_POP          EINSUMS_DO_PRAGMA(GCC diagnostic pop)
#    define EINSUMS_DISABLE_WARNING(warningName) EINSUMS_DO_PRAGMA(GCC diagnostic ignored #warningName)

#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE EINSUMS_DISABLE_WARNING(-Wreturn-type-c-linkage)
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS EINSUMS_DISABLE_WARNING(-Wdeprecated-declarations)
// other warnings you want to deactivate...

#else
#    define EINSUMS_DISABLE_WARNING_PUSH
#    define EINSUMS_DISABLE_WARNING_POP
#    define EINSUMS_DISABLE_WARNING_RETURN_TYPE_C_LINKAGE
#    define EINSUMS_DISABLE_WARNING_DEPRECATED_DECLARATIONS
// other warnings you want to deactivate...

#endif

/**
 * @def THROWS(...)
 *
 * @brief Marks a function as being able to throw an exception.
 * 
 * This macro hopefully provides similar support to the old `throw()` syntax from C++ or
 * Java's `throws()` property. It can also aid the user in determining what kinds of exceptions
 * to expect from a function.
 * If the argument is empty, it is just like using `noexcept`. Otherwise, it is like `noexcept(false)`,
 * which means that it can throw exceptions. This macro is mostly for documenting code.
 */
#define THROWS(...) noexcept(true __VA_OPT__(== false))
// clang-format on