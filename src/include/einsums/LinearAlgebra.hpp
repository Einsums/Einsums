//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include "einsums/Blas.hpp"
#include "einsums/BlockTensor.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/utility/ComplexTraits.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

// For some stupid reason doxygen can't handle this macro here but it can in other files.
#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)
#else
namespace einsums::linear_algebra {
#endif

/**
 * @brief Computes the square sum of a tensor.
 *
 * returns the values scale_out and sumsq_out such that
 * \f[
 *   (scale_{out}^{2})*sumsq_{out} = a( 1 )^{2} +...+ a( n )^{2} + (scale_{in}^{2})*sumsq_{in},
 * \f]
 *
 * Under the hood the LAPACK routine `lassq` is used.
 *
 * @code
 * NEED TO ADD AN EXAMPLE
 * @endcode
 *
 * @tparam AType The type of the tensor
 * @tparam ADataType The underlying data type of the tensor
 * @tparam ARank The rank of the tensor
 * @param a The tensor to compute the sum of squares for
 * @param scale scale_in and scale_out for the equation provided
 * @param sumsq sumsq_in and sumsq_out for the equation provided
 */
template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
    requires CoreRankTensor<AType<ADataType, ARank>, 1, ADataType>
void sum_square(const AType<ADataType, ARank> &a, RemoveComplexT<ADataType> *scale, RemoveComplexT<ADataType> *sumsq) {
    if constexpr (einsums::detail::IsIncoreRankTiledTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
        *sumsq = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < a.grid_size(0); i++) {
            if (!a.has_block(i)) {
                continue;
            }
            RemoveComplexT<ADataType> out;

            sum_square(a.tile(i), scale, &out);

            *sumsq += out;
        }
    } else {
        LabeledSection0();

        int n    = a.dim(0);
        int incx = a.stride(0);
        blas::lassq(n, a.data(), incx, scale, sumsq);
    }
}

/**
 * @brief General matrix multiplication.
 *
 * Takes two rank-2 tensors ( \p A and \p B ) performs the multiplication and stores the result in to another
 * rank-2 tensor that is passed in ( \p C ).
 *
 * In this equation, \p TransA is op(A) and \p TransB is op(B).
 * @f[
 * C = \alpha \;op(A) \;op(B) + \beta C
 * @f]
 *
 * @code
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto B = einsums::create_random_tensor("B", 3, 3);
 * auto C = einsums::create_tensor("C", 3, 3);
 *
 * einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
 * @endcode
 *
 * @tparam TransA Tranpose A? true or false
 * @tparam TransB Tranpose B? true or false
 * @param alpha Scaling factor for the product of A and B
 * @param A First input tensor
 * @param B Second input tensor
 * @param beta Scaling factor for the output tensor C
 * @param C Output tensor
 * @tparam T the underlying data type
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T, typename U>
    requires requires {
        requires CoreRankTensor<AType<T, Rank>, 2, T>;
        requires CoreRankTensor<BType<T, Rank>, 2, T>;
        requires CoreRankTensor<CType<T, Rank>, 2, T>;
        requires !CoreRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (CoreRankBlockTensor<AType<T, Rank>, 2, T> && CoreRankBlockTensor<BType<T, Rank>, 2, T>);
        requires std::convertible_to<U, T>;
    }
void gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const U beta, CType<T, Rank> *C) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<CType<T, Rank>, Rank, T>) {

        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
            throw std::runtime_error("gemm: Tensors need the same number of blocks.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemm<TransA, TransB>(static_cast<T>(alpha), A.block(i), B.block(i), static_cast<T>(beta), &(C->block(i)));
        }

        return;
    } else if constexpr (einsums::detail::IsIncoreRankTiledTensorV<AType<T, Rank>, Rank, T> &&
                         einsums::detail::IsIncoreRankTiledTensorV<BType<T, Rank>, Rank, T> &&
                         einsums::detail::IsIncoreRankTiledTensorV<CType<T, Rank>, Rank, T>) {
        // Check for compatibility.
        if (C->grid_size(0) != A.grid_size(TransA ? 1 : 0) || C->grid_size(1) != B.grid_size(TransB ? 0 : 1)) {
            throw std::runtime_error("gemm: Output tensor needs to have a compatible tile grid with the inputs.");
        }
        if (A.grid_size(TransA ? 0 : 1) != B.grid_size(TransB ? 1 : 0)) {
            throw std::runtime_error("gemm: Input tensors need to have compatible tile grids.");
        }
        for (int i = 0; i < C->grid_size(0); i++) {
            if (C->tile_size(0)[i] != A.tile_size(TransA ? 1 : 0)[i]) {
                throw std::runtime_error("gemm: Tile sizes need to match between all three tensors.");
            }
        }
        for (int i = 0; i < C->grid_size(1); i++) {
            if (C->tile_size(1)[i] != B.tile_size(TransB ? 0 : 1)[i]) {
                throw std::runtime_error("gemm: Tile sizes need to match between all three tensors.");
            }
        }
        for (int i = 0; i < A.grid_size(TransA ? 0 : 1); i++) {
            if (A.tile_size(TransA ? 0 : 1)[i] != B.tile_size(TransB ? 1 : 0)[i]) {
                throw std::runtime_error("gemm: Tile sizes need to match between all three tensors.");
            }
        }

// For every block in C, do matrix multiplication.
#pragma omp parallel for collapse(2)
        for (int i = 0; i < C->grid_size(0); i++) {
            for (int j = 0; j < C->grid_size(1); j++) {
                // Check to see if C will be modified.
                bool modified = false;
                for (int k = 0; k < A.grid_size(TransA ? 0 : 1); k++) {
                    if constexpr (TransA && TransB) {
                        modified |= A.has_tile(k, i) && B.has_tile(j, k);
                    } else if constexpr (!TransA && TransB) {
                        modified |= A.has_tile(i, k) && B.has_tile(j, k);
                    } else if constexpr (TransA && !TransB) {
                        modified |= A.has_tile(k, i) && B.has_tile(k, j);
                    } else {
                        modified |= A.has_tile(i, k) && B.has_tile(k, j);
                    }
                }

                // If C is modified, then loop through and matrix multiply. Otherwise, scale or delete depending on beta.
                if (modified) {
                    bool created = !C->has_tile(i, j);
                    auto &C_tile = C->tile(i, j);
                    if (beta == U{0.0} || created) {
                        C_tile.zero();
                    } else {
                        C_tile *= beta;
                    }

                    for (int k = 0; k < A.grid_size(TransA ? 0 : 1); k++) {
                        if constexpr (TransA && TransB) {
                            if (A.has_tile(k, i) && B.has_tile(j, k)) {
                                gemm<TransA, TransB>(alpha, A.tile(k, i), B.tile(j, k), U{1.0}, &C_tile);
                            }
                        } else if constexpr (TransA && !TransB) {
                            if (A.has_tile(k, i) && B.has_tile(k, j)) {
                                gemm<TransA, TransB>(alpha, A.tile(k, i), B.tile(k, j), U{1.0}, &C_tile);
                            }
                        } else if constexpr (!TransA && TransB) {
                            if (A.has_tile(i, k) && B.has_tile(j, k)) {
                                gemm<TransA, TransB>(alpha, A.tile(i, k), B.tile(j, k), U{1.0}, &C_tile);
                            }
                        } else {
                            if (A.has_tile(i, k) && B.has_tile(k, j)) {
                                gemm<TransA, TransB>(alpha, A.tile(i, k), B.tile(k, j), U{1.0}, &C_tile);
                            }
                        }
                    }
                } else if(C->has_tile(i, j)) {
                    if (beta == U{0.0}) {
                        C->tiles().erase(std::array<int, 2>{i, j});
                    } else {
                        C->tile(i, j) *= beta;
                    }
                }
            }
        }
    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                         einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks()) {
            gemm<TransA, TransB>(static_cast<T>(alpha), (Tensor<T, 2>)A, (Tensor<T, 2>)B, static_cast<T>(beta), C);
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (int i = 0; i < A.num_blocks(); i++) {
                if (A.block_dim(i) == 0) {
                    continue;
                }
                gemm<TransA, TransB>(static_cast<T>(alpha), A.block(i), B.block(i), static_cast<T>(beta),
                                     &((*C)(A.block_range(i), A.block_range(i))));
            }
        }

        return;
    } else {
        LabeledSection0();

        auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
        auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

        blas::gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, static_cast<T>(alpha), A.data(), lda, B.data(), ldb,
                   static_cast<T>(beta), C->data(), ldc);
    }
}

/**
 * @brief General matrix multiplication. Returns new tensor.
 *
 * Takes two rank-2 tensors performs the multiplication and returns the result
 *
 * @code
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto B = einsums::create_random_tensor("B", 3, 3);
 *
 * auto C = einsums::linear_algebra::gemm<false, false>(1.0, A, B);
 * @endcode
 *
 * @tparam TransA Tranpose A?
 * @tparam TransB Tranpose B?
 * @param alpha Scaling factor for the product of A and B
 * @param A First input tensor
 * @param B Second input tensor
 * @returns resulting tensor
 * @tparam T the underlying data type
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType, size_t Rank,
          typename T, typename U>
    requires requires {
        requires CoreRankTensor<AType<T, Rank>, 2, T>;
        requires CoreRankTensor<BType<T, Rank>, 2, T>;
        requires std::convertible_to<U, T>;
    }
auto gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B) -> Tensor<T, 2> {
    LabeledSection0();

    Tensor<T, 2> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(0) : B.dim(1)};

    gemm<TransA, TransB>(static_cast<T>(alpha), A, B, static_cast<T>(0.0), &C);

    return C;
}

/**
 * @brief General matrix-vector multiplication.
 *
 * This function performs one of the matrix-vector operations
 * \f[
 *    y := alpha*A*z + beta*y\mathrm{,\ or\ }y := alpha*A^{T}*z + beta*y,
 * \f]
 * where alpha and beta are scalars, z and y are vectors and A is an
 * \f$m\f$ by \f$n\f$ matrix.
 *
 * @code
 * NEED TO ADD AN EXAMPLE
 * @endcode
 *
 * @tparam TransA Transpose matrix A? true or false
 * @tparam AType The type of the matrix A
 * @tparam XType The type of the vector z
 * @tparam YType The type of the vector y
 * @tparam ARank The rank of the matrix A
 * @tparam XYRank  The rank of the vectors z and y
 * @tparam T The underlying data type
 * @param alpha Scaling factor for the product of A and z
 * @param A Matrix A
 * @param z Vector z
 * @param beta Scaling factor for the output vector y
 * @param y Output vector y
 */
template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T, typename U>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires CoreRankTensor<XType<T, XYRank>, 1, T>;
        requires CoreRankTensor<YType<T, XYRank>, 1, T>;
        requires std::convertible_to<U, T>; // Make sure the alpha and beta can be converted to T
    }
void gemv(const U alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const U beta, YType<T, XYRank> *y) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            gemv(static_cast<T>(alpha), A.block(i), x(A.block_range(i)), static_cast<T>(beta), &((*y)(A.block_range(i))));
        }

    } else {
        LabeledSection1(fmt::format("<TransA={}>", TransA));
        auto m = A.dim(0), n = A.dim(1);
        auto lda  = A.stride(0);
        auto incx = z.stride(0);
        auto incy = y->stride(0);

        blas::gemv(TransA ? 't' : 'n', m, n, static_cast<T>(alpha), A.data(), lda, z.data(), incx, static_cast<T>(beta), y->data(), incy);
    }
}

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
 *
 * This routines assumes the upper triangle of A is stored. The lower triangle is not referenced.
 *
 * @code
 * // Create tensors A and b.
 * auto A = einsums::create_tensor("A", 3, 3);
 * auto b = einsums::create_tensor("b", 3);
 *
 * // Fill A with the symmetric data.
 * A.vector_data() = einsums::VectorData{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
 *
 * // On exit, A is destroyed and replaced with the eigenvectors.
 * // b is replaced with the eigenvalues in ascending order.
 * einsums::linear_algebra::syev(&A, &b);
 * @endcode
 *
 * @tparam AType The type of the tensor A
 * @tparam ARank The rank of the tensor A (required to be 2)
 * @tparam WType The type of the tensor W
 * @tparam WRank The rank of the tensor W (required to be 1)
 * @tparam T The underlying data type (required to be real)
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param A
 *   On entry, the symmetric matrix A in the leading N-by-N upper triangular part of A.
 *   On exit, if eigenvectors are requested, the orthonormal eigenvectors of A.
 *   Any data previously stored in A is destroyed.
 * @param W On exit, the eigenvalues in ascending order.
 */
template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires CoreRankTensor<WType<T, WRank>, 1, T>;
        requires !Complex<T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            if (A->block_range(i)[1] < 0) {
                printf("Hi");
            }
            auto out_block = (*W)(A->block_range(i));
            syev(&(A->block(i)), &out_block);
        }
    } else {

        LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));

        assert(A->dim(0) == A->dim(1));

        auto           n     = A->dim(0);
        auto           lda   = A->stride(0);
        int            lwork = 3 * n;
        std::vector<T> work(lwork);

        blas::syev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork);
    }
}

template <template <typename, size_t> typename AType, size_t ARank, template <Complex, size_t> typename WType, size_t WRank, typename T,
          bool ComputeLeftRightEigenvectors = true>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires CoreRankTensor<WType<AddComplexT<T>, WRank>, 1, AddComplexT<T>>;
    }
void geev(AType<T, ARank> *A, WType<AddComplexT<T>, WRank> *W, AType<T, ARank> *lvecs, AType<T, ARank> *rvecs) {
    LabeledSection1(fmt::format("<ComputeLeftRightEigenvectors={}>", ComputeLeftRightEigenvectors));

    assert(A->dim(0) == A->dim(1));
    assert(W->dim(0) == A->dim(0));
    assert(A->dim(0) == lvecs->dim(0));
    assert(A->dim(1) == lvecs->dim(1));
    assert(A->dim(0) == rvecs->dim(0));
    assert(A->dim(1) == rvecs->dim(1));

    blas::geev(ComputeLeftRightEigenvectors ? 'v' : 'n', ComputeLeftRightEigenvectors ? 'v' : 'n', A->dim(0), A->data(), A->stride(0),
               W->data(), lvecs->data(), lvecs->stride(0), rvecs->data(), rvecs->stride(0));
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename WType, size_t WRank, typename T,
          bool ComputeEigenvectors = true>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires CoreRankTensor<WType<T, WRank>, 1, T>;
        requires Complex<T>;
    }
void heev(AType<T, ARank> *A, WType<RemoveComplexT<T>, WRank> *W) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            heev(&(A->block(i)), &((*W)(A->block_range(i))));
        }
    } else {

        LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));
        assert(A->dim(0) == A->dim(1));

        auto                           n     = A->dim(0);
        auto                           lda   = A->stride(0);
        int                            lwork = 2 * n;
        std::vector<T>                 work(lwork);
        std::vector<RemoveComplexT<T>> rwork(3 * n);

        blas::heev(ComputeEigenvectors ? 'v' : 'n', 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());
    }
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires CoreRankTensor<BType<T, BRank>, 2, T>;
    }
auto gesv(AType<T, ARank> *A, BType<T, BRank> *B) -> int {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<BType<T, BRank>, BRank, T>) {

        if (A->num_blocks() != B->num_blocks()) {
            throw std::runtime_error("gesv: Tensors need the same number of blocks.");
        }

        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &(B->block(i)));

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;

    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        int info_out = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {

            if (A->block_dim(i) == 0) {
                continue;
            }
            int info = gesv(&(A->block(i)), &((*B)(AllT(), A->block_range(i))));

            info_out |= info;

            if (info != 0) {
                println("gesv: Got non-zero return: %d", info);
            }
        }

        return info_out;
    }

    LabeledSection0();

    auto n   = A->dim(0);
    auto lda = A->dim(0);
    auto ldb = B->dim(1);

    auto nrhs = B->dim(0);

    int                   lwork = n;
    std::vector<blas_int> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
/**
 * Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix.
 *
 * This routines assumes the upper triangle of A is stored. The lower triangle is not referenced.
 *
 * @code
 * // Create tensors A and b.
 * auto A = einsums::create_tensor("A", 3, 3);
 *
 * // Fill A with the symmetric data.
 * A.vector_data() = einsums::VectorData{1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0};
 *
 * // On exit, A is not destroyed. The eigenvectors and eigenvalues are returned in a std::tuple.
 * auto [evecs, evals ] = einsums::linear_algebra::syev(A);
 * @endcode
 *
 * @tparam AType The type of the tensor A
 * @tparam ARank The rank of the tensor A (required to be 2)
 * @tparam T The underlying data type (required to be real)
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param A The symmetric matrix A in the leading N-by-N upper triangular part of A.
 * @return std::tuple<Tensor<T, 2>, Tensor<T, 1>> The eigenvectors and eigenvalues.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T, bool ComputeEigenvectors = true>
    requires CoreRankBlockTensor<AType<T, ARank>, 2, T>
auto syev(const AType<T, ARank> &A) -> std::tuple<BlockTensor<T, 2>, Tensor<T, 1>> {
    LabeledSection0();

    BlockTensor<T, 2> a = A;
    Tensor<T, 1>      w{"eigenvalues", A.dim(0)};

    syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}
#endif

template <template <typename, size_t> typename AType, size_t ARank, typename T, bool ComputeEigenvectors = true>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires !CoreRankBlockTensor<AType<T, ARank>, ARank, T>;
    }
auto syev(const AType<T, ARank> &A) -> std::tuple<Tensor<T, 2>, Tensor<T, 1>> {
    LabeledSection0();

    assert(A.dim(0) == A.dim(1));

    Tensor<T, 2> a = A;
    Tensor<T, 1> w{"eigenvalues", A.dim(0)};

    blas::syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

/**
 * Scales a tensor by a scalar.
 *
 * @code
 * auto A = einsums::create_ones_tensor("A", 3, 3);
 *
 * // A is filled with 1.0
 * einsums::linear_algebra::scale(2.0, &A);
 * // A is now filled with 2.0
 * @endcode
 *
 * @tparam AType The type of the tensor
 * @tparam ARank The rank of the tensor
 * @tparam T The underlying data type
 * @param scale The scalar to scale the tensor by
 * @param A The tensor to scale
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
            if (A->block_dim(i) == 0) {
                continue;
            }
            scale(scale, &(A->block(i)));
        }
    } else {

        LabeledSection0();

        blas::scal(A->dim(0) * A->stride(0), scale, A->data(), 1);
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
void scale_row(size_t row, T scale, AType<T, ARank> *A) {
    LabeledSection0();

    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        blas::scal(A->block_dim(A->block_of(row), 1), scale, A->data(row, 0ul), A->block(A->block_of(row)).stride(1));
    } else {
        blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
    }
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
void scale_column(size_t col, T scale, AType<T, ARank> *A) {
    LabeledSection0();

    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        blas::scal(A->block_dim(A->block_of(col), 1), scale, A->data(0ul, col), A->block(A->block_of(col)).stride(0));
    } else {
        blas::scal(A->dim(0), scale, A->data(0ul, col), A->stride(0));
    }
}

/**
 * @brief Computes the matrix power of a to alpha.  Return a new tensor, does not destroy a.
 *
 * @tparam AType
 * @param a Matrix to take power of
 * @param alpha The power to take
 * @param cutoff Values below cutoff are considered zero.
 *
 * @return std::enable_if_t<std::is_base_of_v<Detail::TensorBase<double, 2>, AType>, AType>
 *
 * TODO This function needs to have a test case implemented.
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto pow(const AType<T, ARank> &a, T alpha, T cutoff = std::numeric_limits<T>::epsilon()) -> AType<T, ARank> {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {
        auto out = AType<T, ARank>(a); // Copy a so that this has the same signature.
        out.set_name("pow result");

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < a.num_blocks(); i++) {
            if (a.block_dim(i) == 0) {
                continue;
            }
            out.block(i) = pow(a.block(i), alpha, cutoff);
        }

        return out;
    } else {
        LabeledSection0();

        assert(a.dim(0) == a.dim(1));

        size_t       n  = a.dim(0);
        Tensor<T, 2> a1 = a;
        Tensor<T, 2> result{"pow result", a.dim(0), a.dim(1)};
        Tensor<T, 1> e{"e", n};

        // Diagonalize
        syev(&a1, &e);

        Tensor<T, 2> a2 = a1;

        // Determine the largest magnitude of the eigenvalues to use as a scaling factor for the cutoff.
        double max_e = std::fabs(e(n - 1)) > std::fabs(e(0)) ? std::fabs(e(n - 1)) : std::fabs(e(0));

        for (size_t i = 0; i < n; i++) {
            if (alpha < 0.0 && std::fabs(e(i)) < cutoff * max_e) {
                e(i) = 0.0;
            } else {
                e(i) = std::pow(e(i), alpha);
                if (!std::isfinite(e(i))) {
                    e(i) = 0.0;
                }
            }

            scale_row(i, e(i), &a2);
        }

        gemm<true, false>(1.0, a2, a1, 0.0, &result);

        return result;
    }
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T>
    requires requires {
        requires CoreRankTensor<AType<T, 1>, 1, T>;
        requires CoreRankTensor<BType<T, 1>, 1, T>;
    }
auto dot(const AType<T, 1> &A, const BType<T, 1> &B) -> T {
    LabeledSection0();

    assert(A.dim(0) == B.dim(0));

    auto result = blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    return result;
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires requires {
        requires CoreRankTensor<AType<T, Rank>, Rank, T>;
        requires CoreRankTensor<BType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {

    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks()) {
            return dot((einsums::Tensor<T, Rank>)A, (einsums::Tensor<T, Rank>)B);
        }

        if (A.ranges() != B.ranges()) {
            return dot((einsums::Tensor<T, Rank>)A, (einsums::Tensor<T, Rank>)B);
        }

        T out{0};

#pragma omp parallel for reduction(+ : out)
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            out += dot(A.block(i), B.block(i));
        }

        return out;

    } else {
        LabeledSection0();

        Dim<1> dim{1};

        for (size_t i = 0; i < Rank; i++) {
            assert(A.dim(i) == B.dim(i));
            dim[0] *= A.dim(i);
        }

        return dot(TensorView<T, 1>(const_cast<AType<T, Rank> &>(A), dim), TensorView<T, 1>(const_cast<BType<T, Rank> &>(B), dim));
    }
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires CoreRankTensor<AType<T, Rank>, Rank, T>;
        requires CoreRankTensor<BType<T, Rank>, Rank, T>;
        requires CoreRankTensor<CType<T, Rank>, Rank, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B, const CType<T, Rank> &C) -> T {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<CType<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C.num_blocks() || B.num_blocks() != C.num_blocks()) {
            return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B, (Tensor<T, Rank>)C);
        }

        if (A.ranges() != B.ranges() || A.ranges() != C.ranges() || B.ranges() != C.ranges()) {
            return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B, (Tensor<T, Rank>)C);
        }

        T out{0};

#pragma omp parallel for reduction(+ : out)
        for (int i = 0; i < A.num_blocks(); i++) {
            if (A.block_dim(i) == 0) {
                continue;
            }
            out += dot(A.block(i), B.block(i), C.block(i));
        }

        return out;

    } else {

        LabeledSection0();

        Dim<1> dim{1};

        for (size_t i = 0; i < Rank; i++) {
            assert(A.dim(i) == B.dim(i) && A.dim(i) == C.dim(i));
            dim[0] *= A.dim(i);
        }

        auto vA = TensorView<T, 1>(const_cast<AType<T, Rank> &>(A), dim);
        auto vB = TensorView<T, 1>(const_cast<BType<T, Rank> &>(B), dim);
        auto vC = TensorView<T, 1>(const_cast<CType<T, Rank> &>(C), dim);

        T result{0};
#pragma omp parallel for reduction(+ : result)
        for (size_t i = 0; i < dim[0]; i++) {
            result += vA(i) * vB(i) * vC(i);
        }
        return result;
    }
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires CoreRankTensor<XType<T, Rank>, Rank, T>;
        requires CoreRankTensor<YType<T, Rank>, Rank, T>;
        requires !CoreRankBlockTensor<YType<T, Rank>, Rank, T> ||
                     (CoreRankBlockTensor<XType<T, Rank>, Rank, T> && CoreRankBlockTensor<YType<T, Rank>, Rank, T>);
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<XType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<YType<T, Rank>, Rank, T>) {
        if (X.num_blocks() != Y->num_blocks()) {
            throw std::runtime_error("axpy: Tensors need to have the same number of blocks.");
        }

        if (X.ranges() != Y->ranges()) {
            throw std::runtime_error("axpy: Tensor blocks need to be compatible.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim() == 0) {
                continue;
            }

            axpy(alpha, X[i], &(Y->block(i)));
        }
    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<XType<T, Rank>, Rank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim() == 0) {
                continue;
            }

            std::array<einsums::Range, Rank> slice;

            slice.fill(X.block_range());

            auto Y_block = std::apply(*Y, slice);

            axpy(alpha, X[i], &Y_block);
        }
    } else {

        LabeledSection0();

        blas::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
    }
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires CoreRankTensor<XType<T, Rank>, Rank, T>;
        requires CoreRankTensor<YType<T, Rank>, Rank, T>;
        requires !CoreRankBlockTensor<YType<T, Rank>, Rank, T> ||
                     (CoreRankBlockTensor<XType<T, Rank>, Rank, T> && CoreRankBlockTensor<YType<T, Rank>, Rank, T>);
    }
void axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<XType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<YType<T, Rank>, Rank, T>) {
        if (X.num_blocks() != Y->num_blocks()) {
            throw std::runtime_error("axpby: Tensors need to have the same number of blocks.");
        }

        if (X.ranges() != Y->ranges()) {
            throw std::runtime_error("axpby: Tensor blocks need to be compatible.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim() == 0) {
                continue;
            }
            axpby(alpha, X[i], beta, Y->block(i));
        }
    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<XType<T, Rank>, Rank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < X.num_blocks(); i++) {
            if (X.block_dim() == 0) {
                continue;
            }

            std::array<einsums::Range, Rank> slice;

            slice.fill(X.block_range());

            auto Y_block = std::apply(*Y, slice);

            axpby(alpha, X[i], beta, &Y_block);
        }
    } else {

        LabeledSection0();

        blas::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
    }
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires requires {
        requires CoreRankTensor<XYType<T, XYRank>, 1, T>;
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
    }
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    LabeledSection0();

    blas::ger(X.dim(0), Y.dim(0), alpha, X.data(), X.stride(0), Y.data(), Y.stride(0), A->data(), A->stride(0));
}

/**
 * @brief Computes the LU factorization of a general m-by-n matrix.
 *
 * The routine computes the LU factorization of a general m-by-n matrix A as
 * \f[
 * A = P*L*U
 * \f]
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements and U is upper triangular. The routine uses
 * partial pivoting, with row interchanges.
 *
 * @tparam TensorType
 * @tparam T
 * @tparam TensorRank
 * @param A
 * @param pivot
 * @return
 */
template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
auto getrf(TensorType<T, TensorRank> *A, std::vector<blas_int> *pivot) -> int {
    LabeledSection0();

    if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
        println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
        pivot->resize(std::min(A->dim(0), A->dim(1)));
    }
    int result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->stride(0), pivot->data());

    if (result < 0) {
        println_warn("getrf: argument {} has an invalid value", -result);
        abort();
    }

    return result;
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize \f$A\f$.
 *
 * @tparam TensorType The type of the tensor
 * @tparam T The underlying data type
 * @tparam TensorRank The rank of the tensor
 * @param A The matrix to invert
 * @param pivot The pivot vector from getrf
 * @return int If 0, the execution is successful.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
auto getri(TensorType<T, TensorRank> *A, const std::vector<blas_int> &pivot) -> int {
    LabeledSection0();

    int result = blas::getri(A->dim(0), A->data(), A->stride(0), pivot.data());

    if (result < 0) {
        println_warn("getri: argument {} has an invalid value", -result);
    }
    return result;
}

/**
 * @brief Inverts a matrix.
 *
 * Utilizes the LAPACK routines getrf and getri to invert a matrix.
 *
 * @tparam TensorType The type of the tensor
 * @tparam T The underlying data type
 * @tparam TensorRank The rank of the tensor
 * @param A Matrix to invert. On exit, the inverse of A.
 */
template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
void invert(TensorType<T, TensorRank> *A) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<TensorType<T, TensorRank>, TensorRank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks; i++) {
            einsums::linear_algebra::invert(&(A->block(i)));
        }
    } else {
        LabeledSection0();

        std::vector<blas_int> pivot(A->dim(0));
        int                   result = getrf(A, &pivot);
        if (result > 0) {
            println_abort("invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed", result,
                          result);
        }

        result = getri(A, pivot);
        if (result > 0) {
            println_abort("invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed", result,
                          result);
        }
    }
}

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
template <SmartPointer SmartPtr>
void invert(SmartPtr *A) {
    LabeledSection0();

    return invert(A->get());
}
#endif

/**
 * @brief Indicates the type of norm to compute.
 */
enum class Norm : char {
    MaxAbs    = 'M', /**< \f$val = max(abs(Aij))\f$, largest absolute value of the matrix A. */
    One       = '1', /**< \f$val = norm1(A)\f$, 1-norm of the matrix A (maximum column sum) */
    Infinity  = 'I', /**< \f$val = normI(A)\f$, infinity norm of the matrix A (maximum row sum) */
    Frobenius = 'F', /**< \f$val = normF(A)\f$, Frobenius norm of the matrix A (square root of sum of squares). */
    //    Two       = 'F'
};

/**
 * @brief Computes the norm of a matrix.
 *
 * Returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real matrix A.
 *
 * @note
 * This function assumes that the matrix is stored in column-major order.
 *
 * @code
 * using namespace einsums;
 *
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto norm = einsums::linear_algebra::norm(einsums::linear_algebra::Norm::One, A);
 * @endcode
 *
 * @tparam AType The type of the matrix
 * @tparam ADataType The underlying data type of the matrix
 * @tparam ARank The rank of the matrix
 * @param norm_type where Norm::One denotes the one norm of a matrix (maximum column sum),
 *   Norm::Infinity denotes the infinity norm of a matrix  (maximum row sum) and
 *   Norm::Frobenius denotes the Frobenius norm of a matrix (square root of sum of
 *   squares). Note that \f$ max(abs(A(i,j))) \f$ is not a consistent matrix norm.
 * @param a The matrix to compute the norm of
 * @return The norm of the matrix
 */

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
    requires CoreRankTensor<AType<ADataType, ARank>, 2, ADataType>
auto norm(Norm norm_type, const AType<ADataType, ARank> &a) -> RemoveComplexT<ADataType> {
    LabeledSection0();

    std::vector<RemoveComplexT<ADataType>> work(4 * a.dim(0), 0.0);
    return blas::lange(static_cast<char>(norm_type), a.dim(0), a.dim(1), a.data(), a.stride(0), work.data());
}

// Uses the original svd function found in lapack, gesvd, request all left and right vectors.
template <template <typename, size_t> typename AType, typename T, size_t ARank>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto svd(const AType<T, ARank> &_A) -> std::tuple<Tensor<T, 2>, Tensor<RemoveComplexT<T>, 1>, Tensor<T, 2>> {
    LabeledSection0();

    DisableOMPThreads const nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m   = A.dim(0);
    size_t n   = A.dim(1);
    size_t lda = A.stride(0);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    U.zero();
    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    S.zero();
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    Vt.zero();
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);
    superb.zero();

    int info = blas::gesvd('A', 'A', m, n, A.data(), lda, S.data(), U.data(), m, Vt.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            println_abort("svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

template <template <typename, size_t> typename AType, typename T, size_t Rank>
    requires CoreRankTensor<AType<T, Rank>, 2, T>
auto svd_nullspace(const AType<T, Rank> &_A) -> Tensor<T, 2> {
    LabeledSection0();

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    blas_int m   = A.dim(0);
    blas_int n   = A.dim(1);
    blas_int lda = A.stride(0);

    auto U = create_tensor<T>("U", m, m);
    zero(U);
    auto S = create_tensor<T>("S", n);
    zero(S);
    auto V = create_tensor<T>("V", n, n);
    zero(V);
    auto superb = create_tensor<T>("superb", std::min(m, n) - 2);

    int info = blas::gesvd('N', 'A', m, n, A.data(), lda, S.data(), U.data(), m, V.data(), n, superb.data());

    if (info != 0) {
        if (info < 0) {
            println_abort("svd: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd: error value {}", info);
        }
    }

    // Determine the rank of the nullspace matrix
    int rank = 0;
    for (int i = 0; i < n; i++) {
        if (S(i) > 1e-12) {
            rank++;
        }
    }

    // println("rank {}", rank);
    auto Vview     = V(Range{rank, V.dim(0)}, All);
    auto nullspace = Tensor(V);

    // Normalize nullspace. LAPACK does not guarentee them to be orthonormal
    for (int i = 0; i < nullspace.dim(0); i++) {
        T sum{0};
        for (int j = 0; j < nullspace.dim(1); j++) {
            sum += std::pow(nullspace(i, j), 2.0);
        }
        sum = std::sqrt(sum);
        scale_row(i, sum, &nullspace);
    }

    return nullspace;
}

enum class Vectors : char { All = 'A', Some = 'S', Overwrite = 'O', None = 'N' };

template <template <typename, size_t> typename AType, typename T, size_t ARank>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto svd_dd(const AType<T, ARank> &_A, Vectors job = Vectors::All) -> std::tuple<Tensor<T, 2>, Tensor<RemoveComplexT<T>, 1>, Tensor<T, 2>> {
    LabeledSection0();

    DisableOMPThreads const nothreads;

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A = _A;

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    // Test if it absolutely necessary to zero out these tensors first.
    auto U = create_tensor<T>("U (stored columnwise)", m, m);
    zero(U);
    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    zero(S);
    auto Vt = create_tensor<T>("Vt (stored rowwise)", n, n);
    zero(Vt);

    int info = blas::gesdd(static_cast<char>(job), static_cast<int>(m), static_cast<int>(n), A.data(), static_cast<int>(n), S.data(),
                           U.data(), static_cast<int>(m), Vt.data(), static_cast<int>(n));

    if (info != 0) {
        if (info < 0) {
            println_abort("svd_a: Argument {} has an invalid parameter\n#2 (m) = {}, #3 (n) = {}, #5 (n) = {}, #8 (m) = {}", -info, m, n, n,
                          m);
        } else {
            println_abort("svd_a: error value {}", info);
        }
    }

    return std::make_tuple(U, S, Vt);
}

template <template <typename, size_t> typename AType, typename T, size_t ARank>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto truncated_svd(const AType<T, ARank> &_A, size_t k) -> std::tuple<Tensor<T, 2>, Tensor<RemoveComplexT<T>, 1>, Tensor<T, 2>> {
    LabeledSection0();

    size_t m = _A.dim(0);
    size_t n = _A.dim(1);

    // Omega Test Matrix
    auto omega = create_random_tensor<T>("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<T, 2> Y("Y", m, k + 5);
    gemm<false, false>(T{1.0}, _A, omega, T{0.0}, &Y);

    Tensor<T, 1> tau("tau", std::min(m, k + 5));
    // Compute QR factorization of Y
    int info1 = blas::geqrf(m, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    if constexpr (!IsComplexV<T>) {
        int info2 = blas::orgqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));
    } else {
        int info2 = blas::ungqr(m, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const T *>(tau.data()));
    }

    // Cast the matrix A into a smaller rank (B)
    Tensor<T, 2> B("B", k + 5, n);
    gemm<true, false>(T{1.0}, Y, _A, T{0.0}, &B);

    // Perform svd on B
    auto [Utilde, S, Vt] = svd_dd(B);

    // Cast U back into full basis
    Tensor<T, 2> U("U", m, k + 5);
    gemm<false, false>(T{1.0}, Y, Utilde, T{0.0}, &U);

    return std::make_tuple(U, S, Vt);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto truncated_syev(const AType<T, ARank> &A, size_t k) -> std::tuple<Tensor<T, 2>, Tensor<T, 1>> {
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        println_abort("Non-square matrix used as input of truncated_syev!");
    }

    size_t n = A.dim(0);

    // Omega Test Matrix
    Tensor<double, 2> omega = create_random_tensor("omega", n, k + 5);

    // Matrix Y = A * Omega
    Tensor<double, 2> Y("Y", n, k + 5);
    gemm<false, false>(1.0, A, omega, 0.0, &Y);

    Tensor<double, 1> tau("tau", std::min(n, k + 5));
    // Compute QR factorization of Y
    blas_int const info1 = blas::geqrf(n, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    blas_int const info2 = blas::orgqr(n, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const double *>(tau.data()));

    Tensor<double, 2> &Q1 = Y;

    // Cast the matrix A into a smaller rank (B)
    // B = Q^T * A * Q
    Tensor<double, 2> Btemp("Btemp", k + 5, n);
    gemm<true, false>(1.0, Q1, A, 0.0, &Btemp);
    Tensor<double, 2> B("B", k + 5, k + 5);
    gemm<false, false>(1.0, Btemp, Q1, 0.0, &B);

    // Create buffer for eigenvalues
    Tensor<double, 1> w("eigenvalues", k + 5);

    // Diagonalize B
    syev(&B, &w);

    // Cast U back into full basis (B is column-major so we need to transpose it)
    Tensor<double, 2> U("U", n, k + 5);
    gemm<false, true>(1.0, Q1, B, 0.0, &U);

    return std::make_tuple(U, w);
}

template <typename T>
inline auto pseudoinverse(const Tensor<T, 2> &A, double tol) -> Tensor<T, 2> {
    LabeledSection0();

    auto [U, S, Vh] = svd_a(A);

    size_t new_dim;
    for (size_t v = 0; v < S.dim(0); v++) {
        T val = S(v);
        if (val > tol)
            scale_column(v, 1.0 / val, &U);
        else {
            new_dim = v;
            break;
        }
    }

    TensorView<T, 2> U_view = U(All, Range{0, new_dim});
    TensorView<T, 2> V_view = Vh(Range{0, new_dim}, All);

    Tensor<T, 2> pinv("pinv", A.dim(0), A.dim(1));
    gemm<false, false>(1.0, U_view, V_view, 0.0, &pinv);

    return pinv;
}

template <typename T>
inline auto solve_continuous_lyapunov(const Tensor<T, 2> &A, const Tensor<T, 2> &Q) -> Tensor<T, 2> {
    LabeledSection0();

    if (A.dim(0) != A.dim(1)) {
        println_abort("solve_continuous_lyapunov: Dimensions of A ({} x {}), do not match", A.dim(0), A.dim(1));
    }
    if (Q.dim(0) != Q.dim(1)) {
        println_abort("solve_continuous_lyapunov: Dimensions of Q ({} x {}), do not match", Q.dim(0), Q.dim(1));
    }
    if (A.dim(0) != Q.dim(0)) {
        println_abort("solve_continuous_lyapunov: Dimensions of A ({} x {}) and Q ({} x {}), do not match", A.dim(0), A.dim(1), Q.dim(0),
                      Q.dim(1));
    }

    size_t n = A.dim(0);

    /// TODO: Break this off into a separate schur function
    // Compute Schur Decomposition of A
    Tensor<T, 2>          R = A; // R is a copy of A
    Tensor<T, 2>          wr("Schur Real Buffer", n, n);
    Tensor<T, 2>          wi("Schur Imaginary Buffer", n, n);
    Tensor<T, 2>          U("Lyapunov U", n, n);
    std::vector<blas_int> sdim(1);
    blas::gees('V', n, R.data(), n, sdim.data(), wr.data(), wi.data(), U.data(), n);

    // Compute F = U^T * Q * U
    Tensor<T, 2> Fbuff = gemm<true, false>(1.0, U, Q);
    Tensor<T, 2> F     = gemm<false, false>(1.0, Fbuff, U);

    // Call the Sylvester Solve
    std::vector<T> scale(1);
    blas::trsyl('N', 'N', 1, n, n, const_cast<const T *>(R.data()), n, const_cast<const T *>(R.data()), n, F.data(), n, scale.data());

    Tensor<T, 2> Xbuff = gemm<false, false>(scale[0], U, F);
    Tensor<T, 2> X     = gemm<false, true>(1.0, Xbuff, U);

    return X;
}

ALIAS_TEMPLATE_FUNCTION(solve_lyapunov, solve_continuous_lyapunov)

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, 2, T>
auto qr(const AType<T, ARank> &_A) -> std::tuple<Tensor<T, 2>, Tensor<T, 1>> {
    LabeledSection0();

    // Copy A because it will be overwritten by the QR call.
    Tensor<T, 2>   A = _A;
    const blas_int m = A.dim(0);
    const blas_int n = A.dim(1);

    Tensor<double, 1> tau("tau", std::min(m, n));
    // Compute QR factorization of Y
    blas_int info = blas::geqrf(m, n, A.data(), n, tau.data());

    if (info != 0) {
        println_abort("{} parameter to geqrf has an illegal value.", -info);
    }

    // Extract Matrix Q out of QR factorization
    // blas_int info2 = blas::orgqr(m, n, tau.dim(0), A.data(), n, const_cast<const double *>(tau.data()));
    return {A, tau};
}

template <typename T>
auto q(const Tensor<T, 2> &qr, const Tensor<T, 1> &tau) -> Tensor<T, 2> {
    const blas_int m = qr.dim(1);
    const blas_int p = qr.dim(0);

    Tensor<T, 2> Q = qr;

    blas_int info = blas::orgqr(m, m, p, Q.data(), m, tau.data());
    if (info != 0) {
        println_abort("{} parameter to orgqr has an illegal value. {} {} {}", -info, m, m, p);
    }

    return Q;
}

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)
