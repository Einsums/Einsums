//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/TensorAlgebra/Backends/DevicePermute.hpp>

#include <librett.h>

// HPTT includes <complex> which defined I as a shorthand for complex values.
// This causes issues with einsums since we define I to be a useable index
// for the user. Undefine the one defined in <complex> here.
#if defined(I)
#    undef I
#endif

namespace einsums::tensor_algebra::detail {

void gpu_permute(int const *perm, int const dim, float const alpha, float const *A, int const *sizeA, float const beta, float *B) {
    librettHandle       plan;
    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(float), stream);

    librettExecute(plan, (void *)A, B);
}

void gpu_permute(int const *perm, int const dim, double const alpha, double const *A, int const *sizeA, double const beta, double *B) {
    librettHandle plan;

    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(double), stream);

    librettExecute(plan, (void *)A, B);
}

void gpu_permute(int const *perm, int const dim, hipFloatComplex const alpha, hipFloatComplex const *A, int const *sizeA,
                 hipFloatComplex const beta, hipFloatComplex *B) {
    librettHandle plan;

    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(hipFloatComplex), stream);

    librettExecute(plan, (void *)A, B);
}

void gpu_permute(int const *perm, int const dim, hipDoubleComplex const alpha, hipDoubleComplex const *A, int const *sizeA,
                 hipDoubleComplex const beta, hipDoubleComplex *B) {
    librettHandle plan;

    librett_gpuStream_t stream = gpu::get_stream();

    librettPlan(&plan, dim, (int *)sizeA, (int *)perm, sizeof(hipDoubleComplex), stream);

    librettExecute(plan, (void *)A, B);
}

} // namespace einsums::tensor_algebra::detail