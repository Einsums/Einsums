//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
#    define EINSUMS_OMP_SIMD         _Pragma("omp simd")
#    define EINSUMS_OMP_PARALLEL     _Pragma("omp parallel")
#    define EINSUMS_OMP_TASK_FOR _Pragma("omp taskloop simd")
#    define EINSUMS_OMP_TASK     _Pragma("omp task")
#else
#    define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#    define EINSUMS_OMP_SIMD
#    define EINSUMS_OMP_PARALLEL _Pragma("omp parallel")
#    define EINSUMS_OMP_TASK_FOR _Pragma("omp taskloop")
#    define EINSUMS_OMP_TASK     _Pragma("omp task")
#endif

#define EINSUMS_ALWAYS_INLINE __attribute__((always_inline)) inline