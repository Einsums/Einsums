//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/_GPUUtils.hpp"

#include "einsums/DeviceSort.hpp"

#if defined(EINSUMS_USE_HPTT)
#    include "librett.h"

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#    if defined(I)
#        undef I
#    endif
#endif

namespace einsums::tensor_algebra::detail {

void gpu_sort(const int *perm, const int dim, const float alpha, const float *A, const int *sizeA, const float beta, float *B) {
    librettHandle       plan;
    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(float), stream);

    librettExecute(plan, (void *)A, B);
}

void gpu_sort(const int *perm, const int dim, const double alpha, const double *A, const int *sizeA, const double beta, double *B) {
    librettHandle plan;

    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(double), stream);

    librettExecute(plan, (void *)A, B);
}

void gpu_sort(const int *perm, const int dim, const hipComplex alpha, const hipComplex *A, const int *sizeA, const hipComplex beta,
              hipComplex *B) {
    librettHandle plan;

    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(hipComplex), stream);

    librettExecute(plan, (void *)A, B);
}

void gpu_sort(const int *perm, const int dim, const hipDoubleComplex alpha, const hipDoubleComplex *A, const int *sizeA,
              const hipDoubleComplex beta, hipDoubleComplex *B) {
    librettHandle plan;

    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(hipDoubleComplex), stream);

    librettExecute(plan, (void *)A, B);
}

} // namespace einsums::tensor_algebra::detail