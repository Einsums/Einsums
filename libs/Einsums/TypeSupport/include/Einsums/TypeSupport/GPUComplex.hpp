//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#ifdef EINSUMS_ADD_COMPLEX_OPERATORS

#    include <hip/hip_complex.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>

__host__ __device__ inline hipFloatComplex operator+(hipFloatComplex first, hipFloatComplex second) {
    return hipCaddf(first, second);
}

__host__ __device__ inline hipFloatComplex operator-(hipFloatComplex first, hipFloatComplex second) {
    return hipCsubf(first, second);
}

__host__ __device__ inline hipFloatComplex operator*(hipFloatComplex first, hipFloatComplex second) {
    return hipCmulf(first, second);
}

__host__ __device__ inline hipFloatComplex operator/(hipFloatComplex first, hipFloatComplex second) {
    return hipCdivf(first, second);
}

__host__ __device__ inline hipDoubleComplex operator+(hipDoubleComplex first, hipDoubleComplex second) {
    return hipCadd(first, second);
}

__host__ __device__ inline hipDoubleComplex operator-(hipDoubleComplex first, hipDoubleComplex second) {
    return hipCsub(first, second);
}

__host__ __device__ inline hipDoubleComplex operator*(hipDoubleComplex first, hipDoubleComplex second) {
    return hipCmul(first, second);
}

__host__ __device__ inline hipDoubleComplex operator/(hipDoubleComplex first, hipDoubleComplex second) {
    return hipCdiv(first, second);
}
#endif
