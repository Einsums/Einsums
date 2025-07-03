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
constexpr TensorImpl<T>::TensorImpl(TensorImpl<T> const &other)
    : data_{other.data_}, rank_{other.rank_}, size_{other.size_}, dims_{other.dims_}, strides_{other.strides_},
      is_contiguous_{other.is_contiguous_} {
    gpu_init();
}

template <typename T>
constexpr TensorImpl<T>::TensorImpl(TensorImpl<T> &&other) noexcept
    : data_{other.data_}, rank_{other.rank_}, size_{other.size_}, dims_{std::move(other.dims_)}, strides_{std::move(other.strides_)},
      is_contiguous_{other.is_contiguous_} {
    other.data_ = nullptr;
    other.rank_ = 0;
    other.size_ = 0;
    other.dims_.clear();
    other.strides_.clear();
    gpu_init();
}

template<typename T>
constexpr TensorImpl<T> &TensorImpl<T>::operator=(TensorImpl<T> const &other) {
    data_ = other.data_;
    rank_ = other.rank_;
    size_ = other.size_;
    dims_.resize(rank_);
    strides_.resize(rank_);

    dims_.assign(other.dims_.cbegin(), other.dims_.cend());
    strides_.assign(other.strides_.cbegin(), other.strides_.cend());
    is_contiguous_ = other.is_contiguous_;
    gpu_init();
}

template<typename T>
constexpr TensorImpl<T> &TensorImpl<T>::operator=(TensorImpl<T> &&other) {
    data_          = other.data_;
    rank_          = other.rank_;
    size_          = other.size_;
    dims_          = std::move(other.dims_);
    strides_       = std::move(other.strides_);
    is_contiguous_ = other.is_contiguous_;

    other.data_ = nullptr;
    other.rank_ = 0;
    other.size_ = 0;
    other.dims_.clear();
    other.strides_.clear();
    gpu_init();
}

template<typename T>
constexpr TensorImpl<T>::~TensorImpl() noexcept {
    data_ = nullptr;
    rank_ = 0;
    size_ = 0;
    dims_.clear();
    strides_.clear();
#ifdef EINSUMS_COMPUTE_CODE
    gpu::GPUMemoryTracker::get_singleton().release_handle(gpu_handle_, true);
#endif
}

// Now the more useful constructors.
template<typename T>
template <ContainerOrInitializer Dims>
constexpr TensorImpl<T>::TensorImpl(T *data, Dims const &dims, bool row_major = false)
    : data_{data}, dims_(dims.begin(), dims.end()), strides_(dims.size()), rank_{dims.size()} {

    size_          = dims_to_strides(dims_, strides_, row_major);
    is_contiguous_ = true;
}

template<typename T>
template <ContainerOrInitializer Dims, ContainerOrInitializer Strides>
constexpr TensorImpl<T>::TensorImpl(T *data, Dims const &dims, Strides const &strides)
    : data_{data}, dims_(dims.cbegin(), dims.cend()), strides_(strides.begin(), strides.end()), rank_{dims.size()}, size_{1} {
    for (int i = 0; i < rank_; i++) {
        size_ *= dims_[i];
    }

    // Check to see if it is contiguous.
    if (strides[0] == 1) {
        size_t expected = 1;
        for (int i = 0; i < rank_; i++) {
            if (strides_[i] != expected) {
                is_contiguous_ = false;
                break;
            }
            expected *= dims_[i];
        }
        is_contiguous_ = true;
    } else if (strides[rank_ - 1] == 1) {
        size_t expected = 1;
        for (int i = rank_ - 1; i >= 0; i--) {
            if (strides_[i] != expected) {
                is_contiguous_ = false;
                break;
            }
            expected *= dims_[i];
        }
        is_contiguous_ = true;
    } else {
        is_contiguous_ = false;
    }
}

} // namespace detail
} // namespace einsums