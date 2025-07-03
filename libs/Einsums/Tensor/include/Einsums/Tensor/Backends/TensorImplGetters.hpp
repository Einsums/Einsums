//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Tensor/Backends/TensorImpl.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>

#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/GPUMemory/GPUAllocator.hpp>
#    include <Einsums/GPUMemory/GPUMemoryTracker.hpp>
#    include <Einsums/GPUMemory/GPUPointer.hpp>
#    include <Einsums/GPUStreams/GPUStreams.hpp>
#    include <Einsums/hipBLAS.hpp>

#    include <hip/hip_common.h>
#    include <hip/hip_complex.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#endif

namespace einsums {
namespace detail {

template <typename T>
constexpr size_t TensorImpl<T>::dim(int i) const {
    int temp = i;
    if (temp < 0) {
        temp += rank_;
    }

    if (temp < 0 || temp >= rank_) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to dim is out of range! Expected between {} and {}, got {}.", -rank_,
                                rank_ - 1, i);
    }

    return dims_[temp];
}

template <typename T>
constexpr size_t TensorImpl<T>::stride(int i) const {
    int temp = i;
    if (temp < 0) {
        temp += rank_;
    }

    if (temp < 0 || temp >= rank_) {
        EINSUMS_THROW_EXCEPTION(std::out_of_range, "The index passed to stride is out of range! Expected between {} and {}, got {}.",
                                -rank_, rank_ - 1, i);
    }

    return strides_[temp];
}

} // namespace detail
} // namespace einsums