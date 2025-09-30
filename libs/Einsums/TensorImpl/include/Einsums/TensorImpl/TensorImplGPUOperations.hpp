//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <type_traits>

#include "Einsums/GPUStreams/GPUStreams.hpp"
#include "Einsums/TypeSupport/GPUCast.hpp"

namespace einsums {
namespace detail {

template <typename T1, typename T2>
__global__ void impl_copy_kernel_all1(int n, T1 const *in, T2 *out) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int work_size = n / num_threads;
    int remaining = n % num_threads;

    T1 const *in_pos  = in + (ptrdiff_t)work_size * thread_id;
    T2       *out_pos = out + (ptrdiff_t)work_size * thread_id;

    if (thread_id < remaining) {
        work_size++;
        in_pos += thread_id;
        out_pos += thread_id;
    } else {
        in_pos += remaining;
        out_pos += remaining;
    }

    for (int i = 0; i < work_size; i++) {
        out_pos[i] = HipCast<T2, T1>::cast(in_pos[i]);
    }
}

template <typename T>
__global__ void impl_copy_kernel_all1(int n, T const *in, T *out) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int work_size = n / (4 * num_threads);
    int remaining = n % (4 * num_threads);

    int work_unvec      = remaining / num_threads;
    int remaining_unvec = remaining % num_threads;

    // Deal with the vectorizable data.
    if (work_size > 0) {
        HIP_vector_type<T, 4> const *in_pos  = ((HIP_vector_type<T, 4> const *)in) + (ptrdiff_t)work_size * thread_id;
        HIP_vector_type<T, 4>       *out_pos = ((HIP_vector_type<T, 4> *)out) + (ptrdiff_t)work_size * thread_id;

        for (int i = 0; i < work_size; i++) {
            out_pos[i] = in_pos[i];
        }
    }

    // Then the unvectorizable data.
    T const *in_pos  = in + (ptrdiff_t)(4 * work_size * num_threads + work_unvec * thread_id);
    T       *out_pos = out + (ptrdiff_t)(4 * work_size * num_threads + work_unvec * thread_id);

    if (thread_id < remaining_unvec) {
        work_unvec++;
        in_pos += thread_id;
        out_pos += thread_id;
    } else {
        in_pos += remaining_unvec;
        out_pos += remaining_unvec;
    }

    for (int i = 0; i < work_unvec; i++) {
        out_pos[i] = in_pos[i];
    }
}

template <typename T1, typename T2>
__global__ void impl_copy_kernel(int n, T1 const *in, int incx, T2 *out, int incy) {
    int thread_id, num_threads;

    get_worker_info(thread_id, num_threads);

    int work_size = n / num_threads;
    int remaining = n % num_threads;

    T1 const *in_pos  = in + (ptrdiff_t)work_size * thread_id * incx;
    T2       *out_pos = out + (ptrdiff_t)work_size * thread_id * incy;

    if (thread_id < remaining) {
        work_size++;
        in_pos += thread_id * incx;
        out_pos += thread_id * incy;
    } else {
        in_pos += remaining * incx;
        out_pos += remaining * incy;
    }

    for (int i = 0; i < work_size; i++) {
        *out_pos = HipCast<T2, T1>::cast(*in_pos);
        in_pos += incx;
        out_pos += incy;
    }
}

template <typename T>
void impl_real_contiguous_gpu(TensorImpl<std::complex<T>> const &in, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::hip::copy(in.size(), static_cast<T const *>(in.get_gpu_pointer()), 2 * in.get_incx(), out.get_gpu_pointer(), out.get_incx());
    } else {
        T const *in_data  = static_cast<T const *>(in.data());
        T       *out_data = out.data();

        auto blocks     = gpu::blocks(in.size());
        auto block_dims = gpu::block_size(in.size());

        impl_copy_kernel<<<block_dims, blocks, 0, gpu::get_stream()>>>(in.size(), static_cast<T const *>(in.get_gpu_pointer()), 2,
                                                                       out.get_gpu_pointer(), 1);

        gpu::stream_wait();
    }
}

template <typename T, Container HardDims, Container InStrides, Container OutStrides>
void impl_real_noncontiguous_vectorable_gpu(int depth, int hard_rank, size_t easy_size, HardDims const &dims, std::complex<T> const *in,
                                            InStrides const &in_strides, size_t twice_inc_in, T *out, OutStrides const &out_strides,
                                            size_t inc_out) {
    // inc_in needs to be multiplied by 2 before entry.
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            blas::hip::copy(easy_size, static_cast<T const *>(in), twice_inc_in, out, inc_out);
        } else {
            T const *in_data  = static_cast<T const *>(in.data());
            T       *out_data = out.data();

            auto blocks     = gpu::blocks(in.size());
            auto block_dims = gpu::block_size(in.size());

            impl_copy_kernel<<<block_dims, blocks, 0, gpu::get_stream()>>>(in.size(), static_cast<T const *>(in.get_gpu_pointer()), 2,
                                                                           out.get_gpu_pointer(), 1);

            gpu::stream_wait();
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_real_noncontiguous_vectorable_gpu(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, twice_inc_in,
                                               out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, Container Dims, Container InStrides, Container OutStrides>
void impl_real_noncontiguous_gpu(int depth, int rank, Dims const &dims, std::complex<T> const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if (depth == rank) {
        *out = std::real(*in);
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_real_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                    out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_real(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not copy two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not copy two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_real_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_real_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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

        impl_real_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, 2 * in_incx, out.data(),
                                           out_strides, out_incx);
    }
}

template <typename T>
void impl_imag_contiguous(TensorImpl<std::complex<T>> const &in, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::copy(in.size(), static_cast<T const *>(in.data()) + 1, 2 * in.get_incx(), out.data(), out.get_incx());
    } else {
        T const     *in_data  = static_cast<T const *>(in.data()) + 1;
        T           *out_data = out.data();
        size_t const incx = 2 * in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] = in_data[i * incx];
        }
    }
}

template <typename T, Container HardDims, Container InStrides, Container OutStrides>
void impl_imag_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, std::complex<T> const *in,
                                        InStrides const &in_strides, size_t twice_inc_in, T *out, OutStrides const &out_strides,
                                        size_t inc_out) {
    // inc_in needs to be multiplied by 2 before entry.
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            blas::copy(easy_size, static_cast<T const *>(in) + 1, twice_inc_in, out, inc_out);
        } else {
            T const *in_data = static_cast<T const *>(in) + 1;
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] = in_data[i * twice_inc_in];
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_imag_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, twice_inc_in,
                                               out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, Container Dims, Container InStrides, Container OutStrides>
void impl_imag_noncontiguous(int depth, int rank, Dims const &dims, std::complex<T> const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if (depth == rank) {
        *out = std::imag(*in);
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_imag_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                    out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_imag(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not copy two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not copy two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_imag_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_imag_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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

        impl_imag_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, 2 * in_incx, out.data(),
                                           out_strides, out_incx);
    }
}

template <typename T, typename TOther>
void impl_abs_contiguous(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    TOther const *in_data  = in.data();
    T            *out_data = out.data();
    size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
    EINSUMS_OMP_PARALLEL_FOR_SIMD
    for (size_t i = 0; i < size; i++) {
        out_data[i * incy] = std::abs(in_data[i * incx]);
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_abs_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, std::complex<T> const *in,
                                       InStrides const &in_strides, size_t inc_in, TOther *out, OutStrides const &out_strides,
                                       size_t inc_out) {
    if (depth == hard_rank) {
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < easy_size; i++) {
            out[i * inc_out] = std::abs(in[i * inc_in]);
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_abs_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, inc_in,
                                              out + i * out_strides[depth], out_strides, inc_out);
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_abs_noncontiguous(int depth, int rank, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                            OutStrides const &out_strides) {
    if (depth == rank) {
        *out = std::abs(*in);
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_abs_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                   out_strides);
        }
    }
}

template <typename T, typename TOther>
void impl_abs(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    LabeledSection0();

    if (in.rank() != out.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not copy two tensors of different ranks!");
    }

    if (in.dims() != out.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not copy two tensors with different sizes!");
    }

    if (in.is_column_major() != out.is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_abs_noncontiguous(0, in.rank(), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_abs_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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

        impl_abs_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, hard_dims, in.data(), in_strides, in_incx, out.data(),
                                          out_strides, out_incx);
    }
}

template <typename T>
void impl_conj_contiguous(TensorImpl<std::complex<T>> &x) {
    if constexpr (blas::IsBlasableV<T>) {
        blas::lacgv(x.size(), x.data(), x.get_incx());
    } else {
        T           *x_data = x.data();
        size_t const incx = x.get_incx(), size = x.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            x_data[i * incx] = std::conj(x_data[i * incx]);
        }
    }
}

template <typename T, Container HardDims, Container InStrides>
void impl_conj_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, std::complex<T> *x,
                                        InStrides const &x_strides, size_t incx) {
    // inc_in needs to be multiplied by 2 before entry.
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T>) {
            blas::lacgv(easy_size, x, incx);
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                x[i * incx] = std::conj(x[i * incx]);
            }
        }
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_conj_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, x + i * x_strides[depth], x_strides, incx);
        }
    }
}

template <typename T>
void impl_conj(TensorImpl<T> &x) {
    LabeledSection0();

    if constexpr (!IsComplexV<T>) {
        return;
    } else {
        if (x.is_totally_vectorable()) {
            EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using lacgv.");

            impl_conj_contiguous(x);
        } else {
            EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over lacgv.");

            size_t               easy_size, easy_rank, incx, hard_size;
            BufferVector<size_t> hard_dims, x_strides;

            x.query_vectorable_params(&easy_size, &hard_size, &easy_rank, &incx);

            hard_dims.resize(x.rank() - easy_rank);

            if (x.stride(0) < x.stride(-1)) {
                x_strides.resize(x.rank() - easy_rank);

                for (int i = 0; i < x.rank() - easy_rank; i++) {
                    x_strides[i] = x.stride(i + easy_rank);
                    hard_dims[i] = x.dim(i + easy_rank);
                }
            } else {
                x_strides.resize(x.rank() - easy_rank);

                for (int i = 0; i < x.rank() - easy_rank; i++) {
                    x_strides[i] = x.stride(i);
                    hard_dims[i] = x.dim(i);
                }
            }

            impl_conj_noncontiguous_vectorable(0, x.rank() - easy_rank, easy_size, hard_dims, x.data(), x_strides, incx);
        }
    }
}

template <typename T, typename TOther>
void impl_axpy_contiguous(T alpha, TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
        blas::axpy(in.size(), alpha, in.data(), in.get_incx(), out.data(), out.get_incx());
    } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        TOther const *in_data  = in.data();
        T            *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] += alpha * convert<TOther, T>(in_data[i * incx]);
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_axpy_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, T alpha, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
            blas::axpy(easy_size, alpha, in, inc_in, out, inc_out);
        } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
            EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                    "Can not convert complex to real! Please extract the components you want to use before operating.");
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] += alpha * convert<TOther, T>(in[i * inc_in]);
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (depth == rank) {
            *out += alpha * convert<TOther, T>(*in);
        } else {
            for (int i = 0; i < dims[depth]; i++) {
                impl_axpy_noncontiguous(depth + 1, rank, alpha, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                        out_strides);
            }
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

        impl_axpy_noncontiguous(0, in.rank(), static_cast<T>(alpha), in.dims(), in.data(), in.strides(), out.data(), out.strides());
    } else if (in.is_totally_vectorable() && out.is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using axpy.");

        impl_axpy_contiguous(static_cast<T>(alpha), in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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

        impl_axpy_noncontiguous_vectorable(0, in.rank() - easy_rank, easy_size, static_cast<T>(alpha), hard_dims, in.data(), in_strides,
                                           in_incx, out.data(), out_strides, out_incx);
    }
}

template <typename T, typename TOther>
void impl_scal_contiguous(TOther alpha, TensorImpl<T> &out) {
    if constexpr (blas::IsBlasableV<T> && std::is_same_v<RemoveComplexT<T>, RemoveComplexT<TOther>> &&
                  !(IsComplexV<TOther> && !IsComplexV<T>)) {
        blas::scal(out.size(), alpha, out.data(), out.get_incx());
    } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        T           *out_data = out.data();
        size_t const size = out.size(), incx = out.get_incx();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incx] *= convert<TOther, T>(alpha);
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container OutStrides>
void impl_scal_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, TOther alpha, HardDims const &dims, T *out,
                                        OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (blas::IsBlasableV<T> && std::is_same_v<RemoveComplexT<T>, RemoveComplexT<TOther>> &&
                      !(IsComplexV<TOther> && !IsComplexV<T>)) {
            blas::scal(easy_size, alpha, out, inc_out);
        } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
            EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                    "Can not convert complex to real! Please extract the components you want to use before operating.");
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] *= convert<TOther, T>(alpha);
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

        impl_scal_contiguous(static_cast<T>(alpha), out);
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

        impl_scal_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, static_cast<T>(alpha), hard_dims, out.data(), out_strides,
                                           out_incx);
    }
}

template <typename T, typename TOther>
void impl_div_scalar_contiguous(TOther alpha, TensorImpl<T> &out) {
    if constexpr (std::is_same_v<RemoveComplexT<T>, RemoveComplexT<TOther>> && !(IsComplexV<TOther> && !IsComplexV<T>) &&
                  blas::IsBlasableV<T>) {
        blas::rscl(out.size(), alpha, out.data(), out.get_incx());
    } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        T           *out_data = out.data();
        size_t const size = out.size(), incx = out.get_incx();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incx] /= convert<TOther, T>(alpha);
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container OutStrides>
void impl_div_scalar_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, TOther alpha, HardDims const &dims, T *out,
                                              OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (std::is_same_v<RemoveComplexT<T>, RemoveComplexT<TOther>> && !(IsComplexV<TOther> && !IsComplexV<T>) &&
                      blas::IsBlasableV<T>) {
            blas::rscl(easy_size, alpha, out, inc_out);
        } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
            EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                    "Can not convert complex to real! Please extract the components you want to use before operating.");
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] /= convert<TOther, T>(alpha);
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

        impl_div_scalar_contiguous(static_cast<T>(alpha), out);
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

        impl_div_scalar_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, static_cast<T>(alpha), hard_dims, out.data(),
                                                 out_strides, out_incx);
    }
}

template <typename T, typename TOther>
void impl_mult_contiguous(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        TOther const *in_data  = in.data();
        T            *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] *= convert<TOther, T>(in_data[i * incx]);
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_mult_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (depth == hard_rank) {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] *= convert<TOther, T>(in[i * inc_in]);
            }
        } else {
            for (int i = 0; i < dims[depth]; i++) {
                impl_mult_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, in + i * in_strides[depth], in_strides, inc_in,
                                                   out + i * out_strides[depth], out_strides, inc_out);
            }
        }
    }
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_mult_noncontiguous(int depth, int rank, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                             OutStrides const &out_strides) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (depth == rank) {
            *out *= convert<TOther, T>(*in);
        } else {
            for (int i = 0; i < dims[depth]; i++) {
                impl_mult_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                        out_strides);
            }
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

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        TOther const *in_data  = in.data();
        T            *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] /= convert<TOther, T>(in_data[i * incx]);
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_div_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, TOther const *in,
                                       InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
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
}

template <typename T, typename TOther, Container Dims, Container InStrides, Container OutStrides>
void impl_div_noncontiguous(int depth, int rank, Dims const &dims, TOther const *in, InStrides const &in_strides, T *out,
                            OutStrides const &out_strides) {
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (depth == rank) {
            *out /= convert<TOther, T>(*in);
        } else {
            for (int i = 0; i < dims[depth]; i++) {
                impl_div_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                       out_strides);
            }
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
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using a flat loop.");

        impl_div_contiguous(in, out);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over axpy.");

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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
    } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        TOther const *in_data  = in.data();
        T            *out_data = out.data();
        size_t const  incx = in.get_incx(), incy = out.get_incx(), size = in.size();
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < size; i++) {
            out_data[i * incy] = convert<TOther, T>(in_data[i * incx]);
        }
    }
}

template <typename T, typename TOther, Container HardDims, Container InStrides, Container OutStrides>
void impl_copy_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, TOther const *in,
                                        InStrides const &in_strides, size_t inc_in, T *out, OutStrides const &out_strides, size_t inc_out) {
    if (depth == hard_rank) {
        if constexpr (std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<TOther>> && blas::IsBlasableV<T>) {
            blas::copy(easy_size, in, inc_in, out, inc_out);
        } else if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
            EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                    "Can not convert complex to real! Please extract the components you want to use before operating.");
        } else {
            EINSUMS_OMP_PARALLEL_FOR_SIMD
            for (size_t i = 0; i < easy_size; i++) {
                out[i * inc_out] = convert<TOther, T>(in[i * inc_in]);
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (depth == rank) {
            *out = convert<TOther, T>(*in);
        } else {
            for (int i = 0; i < dims[depth]; i++) {
                impl_copy_noncontiguous(depth + 1, rank, dims, in + i * in_strides[depth], in_strides, out + i * out_strides[depth],
                                        out_strides);
            }
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

        size_t easy_size, in_easy_size, out_easy_size, in_hard_size, out_hard_size, easy_rank, in_easy_rank, out_easy_rank, in_incx,
            out_incx;
        BufferVector<size_t> hard_dims, in_strides, out_strides;

        in.query_vectorable_params(&in_easy_size, &in_hard_size, &in_easy_rank, &in_incx);
        out.query_vectorable_params(&out_easy_size, &out_hard_size, &out_easy_rank, &out_incx);

        if (in_easy_rank < out_easy_rank) {
            easy_rank = in_easy_rank;
            easy_size = in_easy_size;
        } else {
            easy_rank = out_easy_rank;
            easy_size = out_easy_size;
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
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        LabeledSection0();

        if (out.is_totally_vectorable()) {
            EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

            impl_scalar_add_contiguous(convert<U, T>(alpha), out);
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

            impl_scalar_add_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, convert<U, T>(alpha), hard_dims, out.data(),
                                                     out_strides, out_incx);
        }
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
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        LabeledSection0();

        if (out.is_totally_vectorable()) {
            EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using scal.");

            impl_scalar_copy_contiguous(convert<U, T>(alpha), out);
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

            impl_scalar_copy_noncontiguous_vectorable(0, out.rank() - easy_rank, easy_size, convert<U, T>(alpha), hard_dims, out.data(),
                                                      out_strides, out_incx);
        }
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() += *in.data();
            } else {
                impl_scalar_add(*in.data(), out);
            }
        } else {
            impl_axpy(convert<double, T>(1.0), in, out);
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() -= *in.data();
            } else {
                impl_scalar_add(-*in.data(), out);
            }
        } else {
            impl_axpy(convert<double, T>(-1.0), in, out);
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() *= *in.data();
            } else {
                impl_scal(*in.data(), out);
            }
        } else {
            impl_mult(in, out);
        }
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() /= *in.data();
            } else {
                impl_div_scalar(*in.data(), out);
            }
        } else {
            impl_div(in, out);
        }
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
    if constexpr (IsComplexV<TOther> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (in.rank() == 0) {
            if (out.rank() == 0) {
                *out.data() = *in.data();
            } else {
                impl_scalar_copy(*in.data(), out);
            }
        } else {
            impl_copy(in, out);
        }
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
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() += in;
        } else {
            impl_scalar_add(in, out);
        }
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
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() -= in;
        } else {
            impl_scalar_add(-in, out);
        }
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
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() *= in;
        } else {
            impl_scal(in, out);
        }
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
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() /= in;
        } else {
            impl_div_scalar(in, out);
        }
    }
}

/**
 * @brief Fill the output with a value.
 *
 * @param in The value to set.
 * @param out The output data.
 */
template <typename T, typename U>
void copy_to(U in, TensorImpl<T> &out) {
    if constexpr (IsComplexV<U> && !IsComplexV<T>) {
        EINSUMS_THROW_EXCEPTION(complex_conversion_error,
                                "Can not convert complex to real! Please extract the components you want to use before operating.");
    } else {
        if (out.rank() == 0) {
            *out.data() = in;
        } else {
            impl_scalar_copy(in, out);
        }
    }
}

template <typename T>
void copy_real(TensorImpl<std::complex<T>> const &in, TensorImpl<T> &out) {
    if (out.rank() == 0 && in.rank() == 0) {
        *out.data() = std::real(*in.data());
    } else if (in.rank() == 0) {
        copy_to(std::real(*in.data()), out);
    } else {
        impl_real(in, out);
    }
}

template <typename T>
void copy_imag(TensorImpl<std::complex<T>> const &in, TensorImpl<T> &out) {
    if (out.rank() == 0 && in.rank() == 0) {
        *out.data() = std::imag(*in.data());
    } else if (in.rank() == 0) {
        copy_to(std::imag(*in.data()), out);
    } else {
        impl_imag(in, out);
    }
}

template <typename T, typename TOther>
void copy_abs(TensorImpl<TOther> const &in, TensorImpl<T> &out) {
    if (out.rank() == 0 && in.rank() == 0) {
        *out.data() = std::abs(*in.data());
    } else if (in.rank() == 0) {
        copy_to(std::abs(*in.data()), out);
    } else {
        impl_bas(in, out);
    }
}

} // namespace detail
} // namespace einsums