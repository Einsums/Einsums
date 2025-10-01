//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/BLASVendor.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/format.h>

#ifdef EINSUMS_COMPUTE_CODE
#    include <Einsums/hipBLAS.hpp>
#endif

namespace einsums {
namespace linear_algebra {
namespace detail {

template <typename AType, typename BType, typename CType>
void impl_direct_product_contiguous(CType alpha, einsums::detail::TensorImpl<AType> const &a, einsums::detail::TensorImpl<BType> const &b,
                                    CType beta, einsums::detail::TensorImpl<CType> *c) {
    if constexpr (std::is_same_v<AType, BType> && std::is_same_v<AType, CType> && blas::IsBlasableV<AType>) {
        blas::scal(c->size(), beta, c->data(), c->get_incx());
        blas::dirprod(a.size(), alpha, a.data(), a.get_incx(), b.data(), b.get_incx(), c->data(), c->get_incx());
    } else {
        AType const *a_data = a.data();
        BType const *b_data = b.data();
        CType       *c_data = c->data();

        size_t const inca = a.get_incx(), incb = b.get_incx(), incc = c->get_incx(), elems = a.size();

        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < elems; i++) {
            c_data[i * incc] = c_data[i * incc] * beta + alpha * a_data[i * inca] * b_data[i * incb];
        }
    }
}

template <typename AType, typename BType, typename CType, Container HardDims, Container AStrides, Container BStrides, Container CStrides>
void impl_direct_product_noncontiguous_vectorable(int depth, int hard_rank, size_t easy_size, HardDims const &dims, CType alpha,
                                                  AType const *a_data, AStrides const &a_strides, size_t inca, BType const *b_data,
                                                  BStrides const &b_strides, size_t incb, CType beta, CType *c_data,
                                                  CStrides const &c_strides, size_t incc) {
    if (depth == hard_rank) {
        blas::scal(easy_size, beta, c_data, incc);
        blas::dirprod(easy_size, alpha, a_data, inca, b_data, incb, c_data, incc);
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_direct_product_noncontiguous_vectorable(depth + 1, hard_rank, easy_size, dims, alpha, a_data + i * a_strides[depth],
                                                         a_strides, inca, b_data + i * b_strides[depth], b_strides, incb, beta,
                                                         c_data + i * c_strides[depth], c_strides, incc);
        }
    }
}

template <typename AType, typename BType, typename CType, Container HardDims, Container AStrides, Container BStrides, Container CStrides>
void impl_direct_product_noncontiguous(int depth, int rank, HardDims const &dims, CType alpha, AType const *a_data,
                                       AStrides const &a_strides, BType const *b_data, BStrides const &b_strides, CType beta, CType *c_data,
                                       CStrides const &c_strides) {
    if (depth == rank) {
        *c_data = beta * *c_data + alpha * *a_data * *b_data;
    } else {
        for (int i = 0; i < dims[depth]; i++) {
            impl_direct_product_noncontiguous(depth + 1, rank, dims, alpha, a_data + i * a_strides[depth], a_strides,
                                              b_data + i * b_strides[depth], b_strides, beta, c_data + i * c_strides[depth], c_strides);
        }
    }
}

template <typename AType, typename BType, typename CType>
void impl_direct_product(CType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B, CType beta,
                         einsums::detail::TensorImpl<CType> *C) {
    LabeledSection0();

    if (A.rank() != B.rank() || A.rank() != C->rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can not combine tensors of different ranks!");
    }

    if (A.dims() != B.dims() || A.dims() != C->dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can not combine tensors with different sizes!");
    }

#ifdef EINSUMS_COMPUTE_CODE
    if constexpr (std::is_same_v<AType, BType> && std::is_same_v<AType, CType> && blas::IsBlasableV<AType>) {
        gpu::GPUAllocator<AType> A_alloc;
        gpu::GPUAllocator<BType> B_alloc;
        gpu::GPUAllocator<CType> C_alloc;

        size_t buffer_size = std::min(A_alloc.work_size(), std::min(B_alloc.work_size(), C_alloc.work_size()));

        if (A.size() > buffer_size) {
            // Calculate the things needed to loop over the tensors.
            size_t    loop_step = 1, loop_skip = 0;
            ptrdiff_t rank_step = -1, rank_skip = -1;
            size_t    view1_size = 1, remaining_size = 0, step_size = 1;
            bool      found_max = false;

            for (int i = A.rank() - 1; i >= 0; i--) {
                if (buffer_size > A.dim(i) * view1_size && !found_max) {
                    view1_size *= A.dim(i);
                } else if (buffer_size <= A.dim(i) * view1_size && view1_size < buffer_size && !found_max) {
                    size_t max_dim = buffer_size / view1_size;
                    rank_skip      = i;
                    view1_size *= max_dim;
                    step_size      = max_dim;
                    remaining_size = A.dim(i) % max_dim;
                    loop_skip      = A.dim(i) / max_dim;
                    found_max      = true;
                } else {
                    loop_step *= A.dim(i);
                    rank_step = std::max(rank_step, (ptrdiff_t)i);
                }
            }

            if (rank_skip < rank_step) {
                rank_skip = rank_step;
            }

            // Set up the indices for the view.
            BufferVector<Range> view_indices(A.rank());

            for (int i = rank_skip; i < A.rank(); i++) {
                view_indices[i] = Range{0, A.dim(i)};
            }

            auto A_block = BlockManager::get_singleton().request_gpu_block(buffer_size * sizeof(AType));
            auto A_lock  = A_block.lock();
            auto B_block = BlockManager::get_singleton().request_gpu_block(buffer_size * sizeof(BType));
            auto B_lock  = B_block.lock();
            auto C_block = BlockManager::get_singleton().request_gpu_block(buffer_size * sizeof(CType));
            auto C_lock  = C_block.lock();

            for (size_t i = 0; i < loop_step; i++) {
                size_t temp = i;
                for (int k = rank_step; k >= 0; k--) {
                    view_indices[k] = Range{temp % A.dim(k), temp % A.dim(k) + 1};
                    temp /= A.dim(k);
                }
                for (size_t j = 0; j < loop_skip; j++) {
                    // Generate the view.
                    view_indices[rank_skip] = Range{j * step_size, (j + 1) * step_size};

                    // Find the view.
                    auto A_view = A.subscript(view_indices);
                    auto B_view = B.subscript(view_indices);
                    auto C_view = C->subscript(view_indices);

                    // Make the view point to the buffer.
                    A_view.set_gpu_memory(A_lock);
                    B_view.set_gpu_memory(B_lock);
                    C_view.set_gpu_memory(C_lock);

                    // Copy the data to the GPU.
                    A_view.tensor_to_gpu();
                    B_view.tensor_to_gpu();
                    C_view.tensor_to_gpu();

                    // Perform the operation.
                    blas::gpu::scal(C_view.size(), beta, C_view.get_gpu_pointer().get(), 1);
                    blas::gpu::dirprod(A_view.size(), alpha, A_view.get_gpu_pointer().get(), 1, B_view.get_gpu_pointer().get(), 1,
                                       C_view.get_gpu_pointer().get(), 1);
                }

                // Handle the remainder.
                if (remaining_size != 0) {
                    view_indices[rank_skip] = Range{loop_skip * step_size, loop_skip * step_size + remaining_size};

                    // Find the view.
                    auto A_view = A.subscript(view_indices);
                    auto B_view = B.subscript(view_indices);
                    auto C_view = C->subscript(view_indices);

                    // Make the view point to the buffer.
                    A_view.set_gpu_memory(A_lock);
                    B_view.set_gpu_memory(B_lock);
                    C_view.set_gpu_memory(C_lock);

                    // Copy the data to the GPU.
                    A_view.tensor_to_gpu();
                    B_view.tensor_to_gpu();
                    C_view.tensor_to_gpu();

                    // Perform the operation.
                    blas::gpu::scal(C_view.size(), beta, C_view.get_gpu_pointer().get(), 1);
                    blas::gpu::dirprod(A_view.size(), alpha, A_view.get_gpu_pointer().get(), 1, B_view.get_gpu_pointer().get(), 1,
                                       C_view.get_gpu_pointer().get(), 1);
                }
            }
            return;
        }
    }
#endif

    if (A.is_column_major() != B.is_column_major() || A.is_column_major() != C->is_column_major()) {
        EINSUMS_LOG_DEBUG("Can't necessarily combine row major and column major tensors. Using the fallback algorithm.");

        impl_direct_product_noncontiguous(0, A.rank(), A.dims(), alpha, A.data(), A.strides(), B.data(), B.strides(), beta, C->data(),
                                          C->strides());
    } else if (A.is_totally_vectorable() && B.is_totally_vectorable() && C->is_totally_vectorable()) {
        EINSUMS_LOG_DEBUG("Inputs were able to be treated as vector inputs and have the same memory layout. Using direct product.");

        impl_direct_product_contiguous(alpha, A, B, beta, C);
    } else {
        EINSUMS_LOG_DEBUG("Inputs were not contiguous, but have the same layout. Using loops over direct product.");

        size_t easy_size, A_easy_size, B_easy_size, C_easy_size, A_hard_size, B_hard_size, C_hard_size, easy_rank, A_easy_rank, B_easy_rank,
            C_easy_rank, inca, incb, incc;
        BufferVector<size_t> hard_dims, A_strides, B_strides, C_strides;

        A.query_vectorable_params(&A_easy_size, &A_hard_size, &A_easy_rank, &inca);
        B.query_vectorable_params(&B_easy_size, &B_hard_size, &B_easy_rank, &incb);
        C->query_vectorable_params(&C_easy_size, &C_hard_size, &C_easy_rank, &incc);

        if (A_easy_rank < B_easy_rank) {
            if (C_easy_rank < A_easy_rank) {
                easy_rank = C_easy_rank;
                easy_size = C_easy_size;
            } else {
                easy_rank = A_easy_rank;
                easy_size = A_easy_size;
            }
        } else {
            if (C_easy_rank < B_easy_rank) {
                easy_rank = C_easy_rank;
                easy_size = C_easy_size;
            } else {
                easy_rank = B_easy_rank;
                easy_size = B_easy_size;
            }
        }

        hard_dims.resize(A.rank() - easy_rank);

        if (A.stride(0) < A.stride(-1)) {
            A_strides.resize(A.rank() - easy_rank);
            B_strides.resize(B.rank() - easy_rank);
            C_strides.resize(C->rank() - easy_rank);

            for (int i = 0; i < A.rank() - easy_rank; i++) {
                A_strides[i] = A.stride(i + easy_rank);
                B_strides[i] = B.stride(i + easy_rank);
                C_strides[i] = C->stride(i + easy_rank);
                hard_dims[i] = A.dim(i + easy_rank);
            }
        } else {
            A_strides.resize(A.rank() - easy_rank);
            B_strides.resize(B.rank() - easy_rank);
            C_strides.resize(C->rank() - easy_rank);

            for (int i = 0; i < A.rank() - easy_rank; i++) {
                A_strides[i] = A.stride(i);
                B_strides[i] = B.stride(i);
                C_strides[i] = C->stride(i);
                hard_dims[i] = A.dim(i);
            }
        }

        impl_direct_product_noncontiguous_vectorable(0, A.rank() - easy_rank, easy_size, hard_dims, alpha, A.data(), A_strides, inca,
                                                     B.data(), B_strides, incb, beta, C->data(), C_strides, incc);
    }
}

#ifndef DOXYGEN
extern template EINSUMS_EXPORT void impl_direct_product<float, float, float>(float alpha, einsums::detail::TensorImpl<float> const &A,
                                                                             einsums::detail::TensorImpl<float> const &B, float beta,
                                                                             einsums::detail::TensorImpl<float> *C);
extern template EINSUMS_EXPORT void impl_direct_product<double, double, double>(double alpha, einsums::detail::TensorImpl<double> const &A,
                                                                                einsums::detail::TensorImpl<double> const &B, double beta,
                                                                                einsums::detail::TensorImpl<double> *C);
extern template EINSUMS_EXPORT void impl_direct_product<std::complex<float>, std::complex<float>, std::complex<float>>(
    std::complex<float> alpha, einsums::detail::TensorImpl<std::complex<float>> const &A,
    einsums::detail::TensorImpl<std::complex<float>> const &B, std::complex<float> beta,
    einsums::detail::TensorImpl<std::complex<float>> *C);
extern template EINSUMS_EXPORT void impl_direct_product<std::complex<double>, std::complex<double>, std::complex<double>>(
    std::complex<double> alpha, einsums::detail::TensorImpl<std::complex<double>> const &A,
    einsums::detail::TensorImpl<std::complex<double>> const &B, std::complex<double> beta,
    einsums::detail::TensorImpl<std::complex<double>> *C);
#endif

} // namespace detail
} // namespace linear_algebra
} // namespace einsums