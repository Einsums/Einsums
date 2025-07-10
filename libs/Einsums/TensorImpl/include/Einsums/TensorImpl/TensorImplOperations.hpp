//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <type_traits>

namespace einsums {
namespace detail {

template <typename T, typename TOther>
void impl_axpy_contiguous(T alpha, TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
        blas::axpy(in.size(), alpha, in.data(), in.get_incx(), out.data(), out.get_incx());
    } else {
        TOther const *in_data  = in.data();
        T            *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] += alpha * in_data[i * incx];
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_axpy_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T alpha, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
            blas::axpy(easy_size, alpha, in, inc_in, out, inc_out);
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] += alpha * in[i * inc_in];
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_axpy_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, alpha, dims, in + i * in_strides[depth], in_strides, inc_in,
                                               out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_axpy_noncontiguous(int depth, int rank, T alpha, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if (depth == rank) {
        *out += alpha * *in;
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_axpy_noncontiguous(depth + 1, rank, alpha, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                    out_strides);
        }
    }
}

template <typename T, typename TOther, typename U>
void impl_axpy(U alpha, TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not add two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not add two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_axpy_noncontiguous(0, in.rank(), (T)alpha, in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_axpy_contiguous((T)alpha, in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, hard_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank,
            in_incx, out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
            hard_size = in_hard_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
            hard_size = out_hard_size;
        }

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i + easy_rank);
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i);
                out_strides[i] = out.stride(i);
                hard_dims[i]   = in.dim(i);
            }
        }

        impl_axpy_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, (T)alpha, hard_dims, in.data(), in_strides, in_incx,
                                           out.data(), out_strides, out_incx);
    }
}

template <typename T>
void impl_scal_contiguous(T alpha, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::scal(out.size(), alpha, out.data(), out.get_incx());
    } else {
        T           *out_data = out.data();
        size_t const size = out.size(), incx = out.get_incx();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incx] *= alpha;
        }
    }
}

template <typename T, Container HardDims, Container OutStrides>
void impl_scal_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T alpha, HardDims const &dims, T *out,
                                        OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            blas::scal(easy_size, alpha, out, inc_out);
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] *= alpha;
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_scal_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, alpha, dims, out + i * out_strides[depth], out_strides,
                                               inc_out);
        }
    }
}

template <typename T, typename U>
void impl_scal(U alpha, TensorImpl<T> &out) {
    LabeledSection0();

    if (out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

        impl_scal_contiguous((T)alpha, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over scal.");

        size_t               easy_size, hard_size, easy_rank, out_incx;
        BufferVector<size_t> hard_dims, out_strides;

        out.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &out_incx);

        hard_dims.resize(out.rank() - easy_rank);

        if (out.stride(0) < out.stride(-1)) {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = out.dim(i + easy_rank);
            }
        } else {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i);
                hard_dims[i]   = out.dim(i);
            }
        }

        impl_scal_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, (T)alpha, hard_dims, out.data(), out_strides, out_incx);
    }
}

template <typename T>
void impl_div_scalar_contiguous(T alpha, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::scal(out.size(), T{1.0} / alpha, out.data(), out.get_incx());
    } else {
        T           *out_data = out.data();
        size_t const size = out.size(), incx = out.get_incx();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incx] /= alpha;
        }
    }
}

template <typename T, Container HardDims, Container OutStrides>
void impl_div_scalar_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T alpha, HardDims const &dims, T *out,
                                              OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            blas::scal(easy_size, T{1.0} / alpha, out, inc_out);
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] /= alpha;
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_div_scalar_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, alpha, dims, out + i * out_strides[depth],
                                                     out_strides, inc_out);
        }
    }
}

template <typename T, typename U>
void impl_div_scalar(U alpha, TensorImpl<T> &out) {
    LabeledSection0();

    if (out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

        impl_div_scalar_contiguous((T)alpha, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over scal.");

        size_t               easy_size, hard_size, easy_rank, out_incx;
        BufferVector<size_t> hard_dims, out_strides;

        out.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &out_incx);

        hard_dims.resize(out.rank() - easy_rank);

        if (out.stride(0) < out.stride(-1)) {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = out.dim(i + easy_rank);
            }
        } else {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i);
                hard_dims[i]   = out.dim(i);
            }
        }

        impl_div_scalar_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, (T)alpha, hard_dims, out.data(), out_strides,
                                                 out_incx);
    }
}

template <typename T, typename TOther>
void impl_mult_contiguous(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    TOther const *in_data  = in.data();
    T            *out_data = out.data();
    size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < size; i++) {
        out_data[i * incy] *= in_data[i * incx];
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_mult_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < easy_size; i++) {
            out[i * inc_out] *= in[i * inc_in];
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_mult_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, inc_in,
                                               out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_mult_noncontiguous(int depth, int rank, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if (depth == rank) {
        *out *= *in;
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_mult_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                    out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_mult(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not multiply two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not multiply two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_mult_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_mult_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, hard_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank,
            in_incx, out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
            hard_size = in_hard_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
            hard_size = out_hard_size;
        }

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i + easy_rank);
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i);
                out_strides[i] = out.stride(i);
                hard_dims[i]   = in.dim(i);
            }
        }

        impl_mult_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx, out.data(),
                                           out_strides, out_incx);
    }
}

template <typename T, typename TOther>
void impl_div_contiguous(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    TOther const *in_data  = in.data();
    T            *out_data = out.data();
    size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < size; i++) {
        out_data[i * incy] /= in_data[i * incx];
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_div_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, TOther const *in,
                                       InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < easy_size; i++) {
            out[i * inc_out] /= in[i * inc_in];
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_div_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, inc_in,
                                              out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_div_noncontiguous(int depth, int rank, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                            OutStrides const &out_strides) {
    if (depth == rank) {
        *out /= *in;
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_div_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                   out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_div(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not divide two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not divide two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_div_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_div_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, hard_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank,
            in_incx, out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
            hard_size = in_hard_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
            hard_size = out_hard_size;
        }

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i + easy_rank);
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i);
                out_strides[i] = out.stride(i);
                hard_dims[i]   = in.dim(i);
            }
        }

        impl_div_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx, out.data(),
                                          out_strides, out_incx);
    }
}

template <typename T, typename TOther>
void impl_copy_contiguous(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
        blas::copy(in.size(), in.data(), in.get_incx(), out.data(), out.get_incx());
    } else {
        TOther const *in_data  = in.data();
        T            *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] = in_data[i * incx];
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_copy_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
            blas::copy(easy_size, in, inc_in, out, inc_out);
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] = in[i * inc_in];
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_copy_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, inc_in,
                                               out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_copy_noncontiguous(int depth, int rank, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if (depth == rank) {
        *out = *in;
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_copy_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                    out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_copy(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not copy two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not copy two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_copy_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_copy_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, hard_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank,
            in_incx, out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
            hard_size = in_hard_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
            hard_size = out_hard_size;
        }

        hard_dims.resize(in.rank() - easy_rank);

        if (in.stride(0) < in.stride(-1)) {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i + easy_rank);
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = in.dim(i + easy_rank);
            }
        } else {
            in_strides.resize(in.rank() - easy_rank);
            out_strides.resize(in.rank() - easy_rank);

            for (int i = 0; i < in.rank() - easy_rank; i++) {
                in_strides[i]  = in.stride(i);
                out_strides[i] = out.stride(i);
                hard_dims[i]   = in.dim(i);
            }
        }

        impl_copy_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx, out.data(),
                                           out_strides, out_incx);
    }
}

template <typename T>
void impl_scalar_add_contiguous(T alpha, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T>) {
        // Use a trick here. If the increment is zero, then it will operate the same value to all outputs.
        blas::axpy(out.size(), T{1.0}, &alpha, 0, out.data(), out.get_incx());
    } else {
        T           *out_data = out.data();
        size_t const incy = out.get_incx(), size = out.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] += alpha;
        }
    }
}

template <typename T, Container HardDims, Container OutStrides>
void impl_scalar_add_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T alpha, HardDims const &dims, T *out,
                                              OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            // Use a trick here. If the increment is zero, then it will operate the same value to all outputs.
            blas::axpy(easy_size, T{1.0}, &alpha, 0, out, inc_out);
        } else {

            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] += alpha;
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_scalar_add_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, alpha, dims, out + i * out_strides[depth],
                                                     out_strides, inc_out);
        }
    }
}

template <typename T, typename U>
void impl_scalar_add(U alpha, TensorImpl<T> &out) {
    LabeledSection0();

    if (out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

        impl_scalar_add_contiguous((T)alpha, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over scal.");

        size_t               easy_size, hard_size, easy_rank, out_incx;
        BufferVector<size_t> hard_dims, out_strides;

        out.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &out_incx);

        hard_dims.resize(out.rank() - easy_rank);

        if (out.stride(0) < out.stride(-1)) {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = out.dim(i + easy_rank);
            }
        } else {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i);
                hard_dims[i]   = out.dim(i);
            }
        }

        impl_scalar_add_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, (T)alpha, hard_dims, out.data(), out_strides,
                                                 out_incx);
    }
}

template <typename T>
void impl_scalar_copy_contiguous(T alpha, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T>) {
        // Use a trick here. If the increment is zero, then it will operate the same value to all outputs.
        blas::copy(out.size(), &alpha, 0, out.data(), out.get_incx());
    } else {
        T           *out_data = out.data();
        size_t const incy = out.get_incx(), size = out.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] = alpha;
        }
    }
}

template <typename T, Container HardDims, Container OutStrides>
void impl_scalar_copy_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T alpha, HardDims const &dims, T *out,
                                               OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            // Use a trick here. If the increment is zero, then it will operate the same value to all outputs.
            blas::copy(easy_size, &alpha, 0, out, inc_out);
        } else {

            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] = alpha;
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_scalar_copy_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, alpha, dims, out + i * out_strides[depth],
                                                      out_strides, inc_out);
        }
    }
}

template <typename T, typename U>
void impl_scalar_copy(U alpha, TensorImpl<T> &out) {
    LabeledSection0();

    if (out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

        impl_scalar_copy_contiguous((T)alpha, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over scal.");

        size_t               easy_size, hard_size, easy_rank, out_incx;
        BufferVector<size_t> hard_dims, out_strides;

        out.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &out_incx);

        hard_dims.resize(out.rank() - easy_rank);

        if (out.stride(0) < out.stride(-1)) {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i + easy_rank);
                hard_dims[i]   = out.dim(i + easy_rank);
            }
        } else {
            out_strides.resize(out.rank() - easy_rank);

            for (int i = 0; i < out.rank() - easy_rank; i++) {
                out_strides[i] = out.stride(i);
                hard_dims[i]   = out.dim(i);
            }
        }

        impl_scalar_copy_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, (T)alpha, hard_dims, out.data(), out_strides,
                                                  out_incx);
    }
}

/**
 * @brief Add the data from the input to the output.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void add_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (in.rank() == 0) {
        if (out.rank() == 0) {
            out.subscript_no_check() += in.subscript_no_check();
        } else {
            impl_scalar_add(in.subscript_no_check(), out);
        }
    } else {
        if constexpr (std::is_integral_v<T>) {
            impl_axpy(T{1}, in, out);
        } else {
            impl_axpy(T{1.0}, in, out);
        }
    }
}

/**
 * @brief Subtract the input data from the output data.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void sub_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (in.rank() == 0) {
        if (out.rank() == 0) {
            out.subscript_no_check() -= in.subscript_no_check();
        } else {
            impl_scalar_add(-in.subscript_no_check(), out);
        }
    } else {
        if constexpr (std::is_integral_v<T>) {
            impl_axpy(T{-1}, in, out);
        } else {
            impl_axpy(T{-1.0}, in, out);
        }
    }
}

/**
 * @brief Multiply the output data by the input data.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void mult_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (in.rank() == 0) {
        if (out.rank() == 0) {
            out.subscript_no_check() *= in.subscript_no_check();
        } else {
            impl_scal(in.subscript_no_check(), out);
        }
    } else {
        impl_mult(in, out);
    }
}

/**
 * @brief Divide the output data by the input data.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void div_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (in.rank() == 0) {
        if (out.rank() == 0) {
            out.subscript_no_check() /= in.subscript_no_check();
        } else {
            impl_div_scalar(in.subscript_no_check(), out);
        }
    } else {
        impl_div(in, out);
    }
}

/**
 * @brief Copy the data from the input to the output.
 *
 * @param in The input data.
 * @param out The output data.
 */
template <typename T, typename TOther>
void copy_to(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (in.rank() == 0) {
        if (out.rank() == 0) {
            out.subscript_no_check() = in.subscript_no_check();
        } else {
            impl_scalar_copy(in.subscript_no_check(), out);
        }
    } else {
        impl_copy(in, out);
    }
}

/**
 * @brief Add a scalar to the output.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void add_assign(U in, TensorImpl<T> &out) {
    if (out.rank() == 0) {
        out.subscript_no_check() += in;
    } else {
        impl_scalar_add(in, out);
    }
}

/**
 * @brief Subtract a scalar from the output.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void sub_assign(U in, TensorImpl<T> &out) {
    if (out.rank() == 0) {
        out.subscript_no_check() -= in;
    } else {
        impl_scalar_add(-in, out);
    }
}

/**
 * @brief Multiply the output by a scalar.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void mult_assign(U in, TensorImpl<T> &out) {
    if (out.rank() == 0) {
        out.subscript_no_check() *= in;
    } else {
        impl_scal(in, out);
    }
}

/**
 * @brief Divide the output by a scalar.
 *
 * @param in The scalar.
 * @param out The output data.
 */
template <typename T, typename U>
void div_assign(U in, TensorImpl<T> &out) {
    if (out.rank() == 0) {
        out.subscript_no_check() /= in;
    } else {
        impl_div_scalar(in, out);
    }
}

/**
 * @brief Fill the output with a value.
 *
 * @param in The value to set.
 * @param out The output data.
 */
template <typename T>
void copy_to(T in, TensorImpl<T> &out) {
    if (out.rank() == 0) {
        out.subscript_no_check() = in;
    } else {
        impl_scalar_copy(in, out);
    }
}

} // namespace detail
} // namespace einsums