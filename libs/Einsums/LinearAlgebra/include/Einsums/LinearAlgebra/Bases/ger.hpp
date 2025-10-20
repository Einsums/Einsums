//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#ifdef EINSUMS_COMPUTE_CODE
#include <Einsums/hipBLAS.hpp>
#endif

namespace einsums::linear_algebra::detail {

template <typename T>
void impl_ger(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
              einsums::detail::TensorImpl<T> &a) {
    if (a.rank() != 2 || x.rank() != 1 || y.rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to ger need to be rank-1 tensors and the output needs to be a rank-2 tensor.");
    }
    if (x.dim(0) != a.dim(0) || y.dim(0) != a.dim(1)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the tensors passed to ger are incompatible!");
    }

#ifdef EINSUMS_COMPUTE_CODE
    // Check if A is on the GPU. If so, then use the GPU algorithm.
    if (a.get_gpu_pointer()) {
        try {
            auto A_lock = a.gpu_cache_tensor();
            auto X_lock = x.gpu_cache_tensor();
            auto Y_lock = y.gpu_cache_tensor();

            if (a.get_gpu_pointer() && x.get_gpu_pointer() && y.get_gpu_pointer()) {
                blas::gpu::ger(x.dim(0), y.dim(0), alpha, x.get_gpu_pointer().get(), 1, y.get_gpu_pointer().get(), 1,
                               a.get_gpu_pointer().get(), a.dim(0));
                return;
            }
        } catch (std::exception &) {
            // Something happened. Do the CPU algorithm.
        }
    }
#endif

    if (a.is_gemmable()) {
        if (a.is_column_major()) {
            blas::ger(a.dim(0), a.dim(1), alpha, x.data(), x.get_incx(), y.data(), y.get_incy(), a.data(), a.get_lda());
        } else {
            blas::ger(a.dim(1), a.dim(0), alpha, y.data(), y.get_incy(), x.data(), x.get_incx(), a.data(), a.get_lda());
        }
    } else {
        T       *a_data = a.data();
        T const *x_data = x.data();
        T const *y_data = y.data();

        size_t row_stride = a.stride(0), col_stride = a.stride(1), incx = x.get_incx(), incy = y.get_incy(), rows = a.dim(0),
               cols = a.dim(1);

        if (row_stride < col_stride) {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t j = 0; j < cols; j++) {
                for (size_t i = 0; i < rows; i++) {
                    a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                }
            }
        } else {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (size_t i = 0; i < rows; i++) {
                for (size_t j = 0; j < cols; j++) {
                    a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                }
            }
        }
    }
}

template <typename T>
void impl_gerc(T alpha, einsums::detail::TensorImpl<T> const &x, einsums::detail::TensorImpl<T> const &y,
               einsums::detail::TensorImpl<T> &a) {
    if (a.rank() != 2 || x.rank() != 1 || y.rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to ger need to be rank-1 tensors and the output needs to be a rank-2 tensor.");
    }
    if (x.dim(0) != a.dim(0) || y.dim(0) != a.dim(1)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the tensors passed to ger are incompatible!");
    }

#ifdef EINSUMS_COMPUTE_CODE
    // Check if A is on the GPU. If so, then use the GPU algorithm.
    if (a.get_gpu_pointer()) {
        try {
            auto A_lock = a.gpu_cache_tensor();
            auto X_lock = x.gpu_cache_tensor();
            auto Y_lock = y.gpu_cache_tensor();

            if (a.get_gpu_pointer() && x.get_gpu_pointer() && y.get_gpu_pointer()) {
                if constexpr (IsComplexV<T>) {
                    blas::gpu::gerc(x.dim(0), y.dim(0), alpha, x.get_gpu_pointer().get(), 1, y.get_gpu_pointer().get(), 1,
                                    a.get_gpu_pointer().get(), a.dim(0));
                } else {
                    blas::gpu::ger(x.dim(0), y.dim(0), alpha, x.get_gpu_pointer().get(), 1, y.get_gpu_pointer().get(), 1,
                                   a.get_gpu_pointer().get(), a.dim(0));
                }
                return;
            }
        } catch (std::exception &) {
            // Something happened. Do the CPU algorithm.
        }
    }
#endif

    if (a.is_gemmable()) {
        if (a.is_column_major()) {
            blas::gerc(a.dim(0), a.dim(1), alpha, x.data(), x.get_incx(), y.data(), y.get_incy(), a.data(), a.get_lda());
        } else {
            blas::gerc(a.dim(1), a.dim(0), alpha, y.data(), y.get_incy(), x.data(), x.get_incx(), a.data(), a.get_lda());
        }
    } else {
        T       *a_data = a.data();
        T const *x_data = x.data();
        T const *y_data = y.data();

        size_t row_stride = a.stride(0), col_stride = a.stride(1), incx = x.get_incx(), incy = y.get_incy(), rows = a.dim(0),
               cols = a.dim(1);

        if constexpr (IsComplexV<T>) {
            if (row_stride < col_stride) {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t j = 0; j < cols; j++) {
                    for (size_t i = 0; i < rows; i++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * std::conj(y_data[j * incy]);
                    }
                }
            } else {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * std::conj(y_data[j * incy]);
                    }
                }
            }
        } else {
            if (row_stride < col_stride) {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t j = 0; j < cols; j++) {
                    for (size_t i = 0; i < rows; i++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                    }
                }
            } else {
                EINSUMS_OMP_PRAGMA(parallel for collapse(2))
                for (size_t i = 0; i < rows; i++) {
                    for (size_t j = 0; j < cols; j++) {
                        a_data[i * row_stride + j * col_stride] += alpha * x_data[i * incx] * y_data[j * incy];
                    }
                }
            }
        }
    }
}

} // namespace einsums::linear_algebra::detail