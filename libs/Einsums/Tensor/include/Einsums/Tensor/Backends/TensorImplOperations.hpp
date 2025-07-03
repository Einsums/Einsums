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
template <typename TOther>
void TensorImpl<T>::copy_from_both_contiguous(TensorImpl<TOther> const &other) {
    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] = T{std::real(other.data_[i])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[i];
            data_[i] = T{std::real(val), std::imag(val)};
        } else {
            data_[i] = T{other.data_[i]};
        }
    }
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::copy_from(TensorImpl<TOther> const &other) {
    if (other.is_contiguous() && is_contiguous()) {
        copy_from_assume_contiguous(other);
        return;
    }

    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel, other_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] = T{std::real(other.data_[other_sentinel])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val             = other.data_[other_sentinel];
            data_[this_sentinel] = T{std::real(val), std::imag(val)};
        } else {
            data_[this_sentinel] = T{other.data_[other_sentinel]};
        }
    }
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::add_assign_both_contiguous(TensorImpl<TOther> const &other) {
    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    if constexpr (std::is_same_v<TOther, T>) {
#ifdef EINSUMS_COMPUTE_CODE
        auto &singleton = gpu::GPUMemoryTracker::get_singleton();
        if (singleton.handle_is_allocated(gpu_handle_) && singleton.handle_is_allocated(other.gpu_handle_)) {
            auto [this_ptr, test]   = singleton.get_pointer(gpu_handle_, size());
            auto [other_ptr, test2] = singleton.get_pointer(other.gpu_handle_, size());
            auto *one               = gpu::detail::Einsums_GPUMemory_vars::get_singleton().get_const(T{1.0});
            hipblas_catch(hipblasSaxpy(gpu::get_blas_handle(), size(), one, other_ptr, 1, this_ptr, 1));

            gpu::stream_wait();

            copy_from_gpu(this_ptr);

            singleton.release_handle(gpu_handle_);
            singleton.release_handle(other.gpu_handle_);
        } else {
#endif
            blas::axpy(size(), T{1.0}, other.data_, 1, data_, 1);
#ifdef EINSUMS_COMPUTE_CODE
        }
#endif
    } else {

        size_t elems = size();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < elems; i++) {
            if constexpr (!IsComplexV<T>) {
                data_[i] += T{std::real(other.data_[i])};
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
                auto val = other.data_[i];
                data_[i] += T{std::real(val), std::imag(val)};
            } else {
                data_[i] += T{other.data_[i]};
            }
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::add_assign(TensorImpl<TOther> const &other) {
    if (other.is_contiguous() && is_contiguous()) {
        add_assign_both_contiguous(other);
        return;
    }

    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel, other_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] += T{std::real(other.data_[other_sentinel])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[other_sentinel];
            data_[this_sentinel] += T{std::real(val), std::imag(val)};
        } else {
            data_[this_sentinel] += T{other.data_[other_sentinel]};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::sub_assign_both_contiguous(TensorImpl<TOther> const &other) {
    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    if constexpr (std::is_same_v<TOther, T>) {
#ifdef EINSUMS_COMPUTE_CODE
        auto &singleton = gpu::GPUMemoryTracker::get_singleton();
        if (singleton.handle_is_allocated(gpu_handle_) && singleton.handle_is_allocated(other.gpu_handle_)) {
            auto [this_ptr, test]   = singleton.get_pointer(gpu_handle_, size());
            auto [other_ptr, test2] = singleton.get_pointer(other.gpu_handle_, size());
            auto *negative_one      = gpu::detail::Einsums_GPUMemory_vars::get_singleton().get_const(T{-1.0});
            hipblas_catch(hipblasSaxpy(gpu::get_blas_handle(), size(), negative_one, other_ptr, 1, this_ptr, 1));

            gpu::stream_wait();

            copy_from_gpu(this_ptr);

            singleton.release_handle(gpu_handle_);
            singleton.release_handle(other.gpu_handle_);
        } else {
#endif
            blas::axpy(size(), T{-1.0}, other.data_, 1, data_, 1);
#ifdef EINSUMS_COMPUTE_CODE
        }
#endif
    } else {

        size_t elems = size();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < elems; i++) {
            if constexpr (!IsComplexV<T>) {
                data_[i] -= T{std::real(other.data_[i])};
            } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
                auto val = other.data_[i];
                data_[i] -= T{std::real(val), std::imag(val)};
            } else {
                data_[i] -= T{other.data_[i]};
            }
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::sub_assign(TensorImpl<TOther> const &other) {
    if (other.is_contiguous() && is_contiguous()) {
        sub_assign_both_contiguous(other);
        return;
    }

    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel, other_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] -= T{std::real(other.data_[other_sentinel])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[other_sentinel];
            data_[this_sentinel] -= T{std::real(val), std::imag(val)};
        } else {
            data_[this_sentinel] -= T{other.data_[other_sentinel]};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::mul_assign_both_contiguous(TensorImpl<TOther> const &other) {
    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] *= T{std::real(other.data_[i])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[i];
            data_[i] *= T{std::real(val), std::imag(val)};
        } else {
            data_[i] *= T{other.data_[i]};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::mul_assign(TensorImpl<TOther> const &other) {
    if (other.is_contiguous() && is_contiguous()) {
        mul_assign_both_contiguous(other);
        return;
    }

    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel, other_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] *= T{std::real(other.data_[other_sentinel])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[other_sentinel];
            data_[this_sentinel] *= T{std::real(val), std::imag(val)};
        } else {
            data_[this_sentinel] *= T{other.data_[other_sentinel]};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::div_assign_both_contiguous(TensorImpl<TOther> const &other) {
    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] /= T{std::real(other.data_[i])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[i];
            data_[i] /= T{std::real(val), std::imag(val)};
        } else {
            data_[i] /= T{other.data_[i]};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::div_assign(TensorImpl<TOther> const &other) {
    if (other.is_contiguous() && is_contiguous()) {
        div_assign_both_contiguous(other);
        return;
    }

    if (other.rank() != rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only copy data between tensors of the same rank!");
    }

    if (other.dims() != dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only copy data between tensors with the same dimensions!");
    }

    if (other.data() == data() || data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel, other_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel, other.strides(), other_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] /= T{std::real(other.data_[other_sentinel])};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            auto val = other.data_[other_sentinel];
            data_[this_sentinel] /= T{std::real(val), std::imag(val)};
        } else {
            data_[this_sentinel] /= T{other.data_[other_sentinel]};
        }
    }
#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::add_assign_scalar_contiguous(TOther value) {
    if (data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] += T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[i] += T{std::real(value), std::imag(value)};
        } else {
            data_[i] += T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::add_assign_scalar(TOther value) {
    if (is_contiguous()) {
        add_assign_scalar_contiguous(value);
        return;
    }

    if (data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] += T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[this_sentinel] += T{std::real(value), std::imag(value)};
        } else {
            data_[this_sentinel] += T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::sub_assign_scalar_contiguous(TOther value) {
    if (data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] -= T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[i] -= T{std::real(value), std::imag(value)};
        } else {
            data_[i] -= T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::sub_assign_scalar(TOther value) {
    if (is_contiguous()) {
        sub_assign_scalar_contiguous(value);
        return;
    }

    if (data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] -= T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[this_sentinel] -= T{std::real(value), std::imag(value)};
        } else {
            data_[this_sentinel] -= T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::mul_assign_scalar_contiguous(TOther value) {
    if (data() == nullptr) {
        // Don't copy.
        return;
    }
    if constexpr (std::is_same_v<T, TOther>) {
#ifdef EINSUMS_COMPUTE_CODE
        auto &singleton = gpu::GPUMemoryTracker::get_singleton();
        if (singleton.handle_is_allocated(gpu_handle_)) {
            size_t             value_handle = singleton.create_handle();
            gpu::GPUPointer<T> value_ptr    = singleton.get_pointer<T>(value_handle, 1);

            gpu::GPUPointer<T> self_ptr = singleton.get_pointer(gpu_handle_, size());

            hip_catch(hipMemcpy(value_ptr, (void *)&value, sizeof(TOther), hipMemcpyHostToDevice));

            blas::gpu::scal(gpu::get_blas_handle(), size(), (T *)value_ptr, self_ptr, 1);

            singleton.release_handle(value_handle);
            singleton.release_handle(gpu_handle_);
        } else {
#endif
            blas::scal(size(), value, data_, 1);
#ifdef EINSUMS_COMPUTE_CODE
        }
#endif
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] *= T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[i] *= T{std::real(value), std::imag(value)};
        } else {
            data_[i] *= T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::mul_assign_scalar(TOther value) {
    if (is_contiguous()) {
        mul_assign_scalar_contiguous(value);
        return;
    }

    if (data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] *= T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[this_sentinel] *= T{std::real(value), std::imag(value)};
        } else {
            data_[this_sentinel] *= T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::div_assign_scalar_contiguous(TOther value) {
    if (data() == nullptr) {
        // Don't copy.
        return;
    }
    if constexpr (std::is_same_v<T, TOther>) {
#ifdef EINSUMS_COMPUTE_CODE
        auto &singleton = gpu::GPUMemoryTracker::get_singleton();
        if (singleton.handle_is_allocated(gpu_handle_)) {
            size_t             value_handle = singleton.create_handle();
            gpu::GPUPointer<T> value_ptr    = singleton.get_pointer<T>(value_handle, 1);

            gpu::GPUPointer<T> self_ptr = singleton.get_pointer(gpu_handle_, size());

            T temp_val = T{1.0} / value;

            hip_catch(hipMemcpy(value_ptr, (void *)&temp_val, sizeof(TOther), hipMemcpyHostToDevice));

            blas::gpu::scal(gpu::get_blas_handle(), size(), (T *)value_ptr, self_ptr, 1);

            singleton.release_handle(value_handle);
            singleton.release_handle(gpu_handle_);
        } else {
#endif
            blas::scal(size(), T{1.0} / value, data_, 1);
#ifdef EINSUMS_COMPUTE_CODE
        }
#endif
    }

    size_t elems = size();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < elems; i++) {
        if constexpr (!IsComplexV<T>) {
            data_[i] /= T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[i] /= T{std::real(value), std::imag(value)};
        } else {
            data_[i] /= T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

template <typename T>
template <typename TOther>
void TensorImpl<T>::div_assign_scalar(TOther value) {
    if (is_contiguous()) {
        div_assign_scalar_contiguous(value);
        return;
    }

    if (data() == nullptr) {
        // Don't copy.
        return;
    }

    size_t elems = size();

    BufferVector<size_t> index_strides;

    dims_to_strides(dims(), index_strides);

    EINSUMS_OMP_PARALLEL_FOR
    for (size_t i = 0; i < elems; i++) {
        size_t this_sentinel;

        sentinel_to_sentinels(i, index_strides, strides(), this_sentinel);

        if constexpr (!IsComplexV<T>) {
            data_[this_sentinel] /= T{value};
        } else if constexpr (IsComplexV<T> && IsComplexV<TOther> && !std::is_same_v<T, TOther>) {
            data_[this_sentinel] /= T{std::real(value), std::imag(value)};
        } else {
            data_[this_sentinel] /= T{value};
        }
    }

#ifdef EINSUMS_COMPUTE_CODE
    auto &singleton = gpu::GPUMemoryTracker::get_singleton();
    if (singleton.handle_is_allocated(gpu_handle_)) {
        auto [this_ptr, test] = singleton.get_pointer(gpu_handle_, size());
        copy_to_gpu(this_ptr);
        singleton.release_handle(gpu_handle_);
    }
#endif
}

} // namespace detail
} // namespace einsums