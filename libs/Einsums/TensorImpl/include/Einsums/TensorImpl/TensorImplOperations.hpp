//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile/LabeledSection.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

namespace einsums {
namespace detail {

template <typename T, typename TOther>
void impl_axpy_contiguous(T &&alpha, TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>) {
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
void impl_axpy_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T &&alpha, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>) {
            blas::axpy(easy_size, alpha, in, inc_in, out, inc_out);
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] *= in[i * inc_in];
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_axpy_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, std::forward(alpha), dims, in + i * in_strides[depth],
                                               in_strides, inc_in, out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_axpy_noncontiguous(int depth, int rank, T &&alpha, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if (depth == rank) {
        *out += alpha * *in;
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_axpy_noncontiguous(depth + 1, rank, std::forward(alpha), dims, in + i * in_strides[depth], in_strides,
                                    out + i * out_strides[depth], out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_axpy(T &&alpha, TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not add two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not add two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_axpy_noncontiguous(0, in.rank(), std::forward(alpha), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_axpy_contiguous(std::forward(alpha), in, out);
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

        impl_axpy_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, std::forward(alpha), hard_dims, in.data(), in_strides,
                                           in_incx, out.data(), out_strides, out_incx);
    }
}

template <typename T>
void impl_scal_contiguous(T &&alpha, TensorImpl<T> &out) {
    blas::scal(out.size(), alpha, out.data(), out.get_incx());
}

template <typename T, Container HardDims, Container InStrides, Container OutStrides>
void impl_scal_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T &&alpha, HardDims const &dims, T *out,
                                        OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        blas::scal(easy_size, alpha, out, inc_out);
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_scal_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, std::forward(alpha), dims, out + i * out_strides[depth],
                                               out_strides, inc_out);
        }
    }
}

template <typename T>
void impl_scal(T &&alpha, TensorImpl<T> &out) {
    LabeledSection0();

    if (out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

        impl_scal_contiguous(std::forward(alpha), out);
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

        impl_scal_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, std::forward(alpha), hard_dims, out.data(), out_strides,
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
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>) {
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
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>>) {
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
void impl_scalar_add_contiguous(T &&alpha, TensorImpl<T> &out) {
    T           *out_data = out.data();
    size_t const size     = out.size();
    size_t const incx     = out.get_incx();

    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < size; i++) {
        out_data[i * incx] += alpha;
    }
}

template <typename T, Container HardDims, Container InStrides, Container OutStrides>
void impl_scalar_add_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T &&alpha, HardDims const &dims, T *out,
                                              OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < easy_size; i++) {
            out[i * inc_out] += alpha;
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_scalar_add_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, std::forward(alpha), dims,
                                                     out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T>
void impl_scalar_add(T &&alpha, TensorImpl<T> &out) {
    LabeledSection0();

    if (out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

        impl_scalar_add_contiguous(std::forward(alpha), out);
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

        impl_scalar_add_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, std::forward(alpha), hard_dims, out.data(),
                                                 out_strides, out_incx);
    }
}

template <typename T, typename TOther>
void add_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    impl_axpy(T{1.0}, in, out);
}

template <typename T, typename TOther>
void sub_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    impl_axpy(T{-1.0}, in, out);
}

template <typename T, typename TOther>
void mult_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    impl_mult(in, out);
}

template <typename T, typename TOther>
void div_assign(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    impl_div(in, out);
}

template <typename T>
void add_assign(T in, TensorImpl<T> &out) {
    impl_scalar_add(in, out);
}

template <typename T>
void sub_assign(T in, TensorImpl<T> &out) {
    impl_scalar_add(-in, out);
}

template <typename T>
void mult_assign(T in, TensorImpl<T> &out) {
    impl_scal(in, out);
}

template <typename T>
void div_assign(T in, TensorImpl<T> &out) {
    impl_scal(T{1.0} / in, out);
}

} // namespace detail
} // namespace einsums