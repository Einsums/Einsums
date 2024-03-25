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

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)

template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
    requires CoreRankTensor<AType<ADataType, ARank>, 1, ADataType>
void sum_square(const AType<ADataType, ARank> &a, RemoveComplexT<ADataType> *scale, RemoveComplexT<ADataType> *sumsq) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<ADataType, ARank>, ARank, ADataType>) {
        *sumsq = 0;

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < a.num_blocks(); i++) {
            RemoveComplexT<ADataType> out;

            sum_square(a.block(i), scale, &out);

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
 * @brief General matrix multipilication.
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
 * @tparam TransA Tranpose A?
 * @tparam TransB Tranpose B?
 * @param A First input tensor
 * @param B Second input tensor
 * @param C Output tensor
 * @tparam T the underlying data type
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires CoreRankTensor<AType<T, Rank>, 2, T>;
        requires CoreRankTensor<BType<T, Rank>, 2, T>;
        requires CoreRankTensor<CType<T, Rank>, 2, T>;
        requires !CoreRankBlockTensor<CType<T, Rank>, 2, T> ||
                     (CoreRankBlockTensor<AType<T, Rank>, 2, T> && CoreRankBlockTensor<BType<T, Rank>, 2, T>);
    }
void gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const T beta, CType<T, Rank> *C) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T> &&
                  einsums::detail::IsIncoreRankBlockTensorV<CType<T, Rank>, Rank, T>) {

        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C->num_blocks() || B.num_blocks() != C->num_blocks()) {
            throw std::runtime_error("gemm: Tensors need the same number of blocks.");
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            gemm<TransA, TransB>(alpha, A.block(i), B.block(i), beta, &(C->block(i)));
        }

        return;
    } else if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, Rank>, Rank, T> &&
                         einsums::detail::IsIncoreRankBlockTensorV<BType<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks()) {
            gemm<TransA, TransB>(alpha, (Tensor<T, 2>)A, (Tensor<T, 2>)B, beta, C);
        } else {
            EINSUMS_OMP_PARALLEL_FOR
            for (int i = 0; i < A.num_blocks(); i++) {
                gemm<TransA, TransB>(alpha, A.block(i), B.block(i), beta, &((*C)(A.block_range(i), A.block_range(i))));
            }
        }

        return;
    } else {
        LabeledSection0();

        auto m = C->dim(0), n = C->dim(1), k = TransA ? A.dim(0) : A.dim(1);
        auto lda = A.stride(0), ldb = B.stride(0), ldc = C->stride(0);

        blas::gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, C->data(), ldc);
    }
}

/**
 * @brief General matrix multipilication. Returns new tensor.
 *
 * Takes two rank-2 tensors performs the multiplication and returns the result
 *
 * @code
 * auto A = einsums::create_random_tensor("A", 3, 3);
 * auto B = einsums::create_random_tensor("B", 3, 3);
 * auto C = einsums::create_tensor("C", 3, 3);
 *
 * einsums::linear_algebra::gemm<false, false>(1.0, A, B, 0.0, &C);
 * @endcode
 *
 * @tparam TransA Tranpose A?
 * @tparam TransB Tranpose B?
 * @param A First input tensor
 * @param B Second input tensor
 * @returns resulting tensor
 * @tparam T the underlying data type
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType, size_t Rank,
          typename T>
    requires requires {
        requires CoreRankTensor<AType<T, Rank>, 2, T>;
        requires CoreRankTensor<BType<T, Rank>, 2, T>;
    }
auto gemm(const T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B) -> Tensor<T, 2> {
    LabeledSection0();

    Tensor<T, 2> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(0) : B.dim(1)};

    gemm<TransA, TransB>(alpha, A, B, 0.0, &C);

    return C;
}

template <bool TransA, template <typename, size_t> typename AType, template <typename, size_t> typename XType,
          template <typename, size_t> typename YType, size_t ARank, size_t XYRank, typename T>
    requires requires {
        requires CoreRankTensor<AType<T, ARank>, 2, T>;
        requires CoreRankTensor<XType<T, XYRank>, 1, T>;
        requires CoreRankTensor<YType<T, XYRank>, 1, T>;
    }
void gemv(const T alpha, const AType<T, ARank> &A, const XType<T, XYRank> &x, const T beta, YType<T, XYRank> *y) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A.num_blocks(); i++) {
            gemv(alpha, A.block(i), x(A.block_range(i)), beta, &((*y)(A.block_range(i))));
        }

    } else {
        LabeledSection1(fmt::format("<TransA={}>", TransA));
        auto m = A.dim(0), n = A.dim(1);
        auto lda  = A.stride(0);
        auto incx = x.stride(0);
        auto incy = y->stride(0);

        blas::gemv(TransA ? 't' : 'n', m, n, alpha, A.data(), lda, x.data(), incx, beta, y->data(), incy);
    }
}

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
            syev(&(A->block(i)), &((*W)(A->block_range(i))));
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

// This assumes column-major ordering!!
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

    int               lwork = n;
    std::vector<eint> ipiv(lwork);

    int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);
    return info;
}

template <template <typename, size_t> typename AType, size_t ARank, typename T, bool ComputeEigenvectors = true>
    requires CoreRankBlockTensor<AType<T, ARank>, 2, T>
auto syev(const AType<T, ARank> &A) -> std::tuple<BlockTensor<T, 2>, Tensor<T, 1>> {
    LabeledSection0();

    BlockTensor<T, 2> a = A;
    Tensor<T, 1>      w{"eigenvalues", A.dim(0)};

    syev<ComputeEigenvectors>(&a, &w);

    return std::make_tuple(a, w);
}

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

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires CoreRankTensor<AType<T, ARank>, ARank, T>
void scale(T scale, AType<T, ARank> *A) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<AType<T, ARank>, ARank, T>) {

        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks(); i++) {
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

template <template <typename, size_t> typename Type, typename T>
    requires CoreRankTensor<Type<T, 1>, 1, T>
auto dot(const Type<T, 1> &A, const Type<T, 1> &B) -> T {
    LabeledSection0();

    assert(A.dim(0) == B.dim(0));

    auto result = blas::dot(A.dim(0), A.data(), A.stride(0), B.data(), B.stride(0));
    return result;
}

template <template <typename, size_t> typename Type, typename T, size_t Rank>
    requires CoreRankTensor<Type<T, Rank>, Rank, T>
auto dot(const Type<T, Rank> &A, const Type<T, Rank> &B) -> T {

    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<Type<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks()) {
            return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B);
        }

        if (A.ranges() != B.ranges()) {
            return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B);
        }

        T out{0};

#pragma omp parallel for reduction(+ : out)
        for (int i = 0; i < A.num_blocks(); i++) {
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

        return dot(TensorView<T, 1>(const_cast<Type<T, Rank> &>(A), dim), TensorView<T, 1>(const_cast<Type<T, Rank> &>(B), dim));
    }
}

template <template <typename, size_t> typename Type, typename T, size_t Rank>
    requires CoreRankTensor<Type<T, Rank>, Rank, T>
auto dot(const Type<T, Rank> &A, const Type<T, Rank> &B, const Type<T, Rank> &C) -> T {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<Type<T, Rank>, Rank, T>) {
        if (A.num_blocks() != B.num_blocks() || A.num_blocks() != C.num_blocks() || B.num_blocks() != C.num_blocks()) {
            return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B, (Tensor<T, Rank>)C);
        }

        if (A.ranges() != B.ranges() || A.ranges() != C.ranges() || B.ranges() != C.ranges()) {
            return dot((Tensor<T, Rank>)A, (Tensor<T, Rank>)B, (Tensor<T, Rank>)C);
        }

        T out{0};

#pragma omp parallel for reduction(+ : out)
        for (int i = 0; i < A.num_blocks(); i++) {
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

        auto vA = TensorView<T, 1>(const_cast<Type<T, Rank> &>(A), dim);
        auto vB = TensorView<T, 1>(const_cast<Type<T, Rank> &>(B), dim);
        auto vC = TensorView<T, 1>(const_cast<Type<T, Rank> &>(C), dim);

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
    }
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    LabeledSection0();

    blas::axpy(X.dim(0) * X.stride(0), alpha, X.data(), 1, Y->data(), 1);
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires requires {
        requires CoreRankTensor<XType<T, Rank>, Rank, T>;
        requires CoreRankTensor<YType<T, Rank>, Rank, T>;
    }
void axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y) {
    LabeledSection0();

    blas::axpby(X.dim(0) * X.stride(0), alpha, X.data(), 1, beta, Y->data(), 1);
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

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
auto getrf(TensorType<T, TensorRank> *A, std::vector<eint> *pivot) -> int {
    LabeledSection0();

    if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
        println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
        pivot->resize(std::min(A->dim(0), A->dim(1)));
    }
    int result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->stride(0), pivot->data());

    if (result < 0) {
        println("getrf: argument {} has an invalid value", -result);
    std:
        abort();
    }

    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
auto getri(TensorType<T, TensorRank> *A, const std::vector<eint> &pivot) -> int {
    LabeledSection0();

    int result = blas::getri(A->dim(0), A->data(), A->stride(0), pivot.data());

    if (result < 0) {
        println("getri: argument {} has an invalid value", -result);
    }
    return result;
}

template <template <typename, size_t> typename TensorType, typename T, size_t TensorRank>
    requires CoreRankTensor<TensorType<T, TensorRank>, 2, T>
void invert(TensorType<T, TensorRank> *A) {
    if constexpr (einsums::detail::IsIncoreRankBlockTensorV<TensorType<T, TensorRank>, TensorRank, T>) {
        EINSUMS_OMP_PARALLEL_FOR
        for (int i = 0; i < A->num_blocks; i++) {
            invert(&(A->block(i)));
        }
    } else {

        LabeledSection0();

        std::vector<eint> pivot(A->dim(0));
        int               result = getrf(A, &pivot);
        if (result > 0) {
            println("invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed", result,
                    result);
            std::abort();
        }

        result = getri(A, pivot);
        if (result > 0) {
            println("invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed", result,
                    result);
            std::abort();
        }
    }
}

template <SmartPointer SmartPtr>
void invert(SmartPtr *A) {
    LabeledSection0();

    return invert(A->get());
}

enum class Norm : char { MaxAbs = 'M', One = 'O', Infinity = 'I', Frobenius = 'F', Two = 'F' };

// TODO: find the best way to find the norm of a block tensor, especially for Frobenius.
template <template <typename, size_t> typename AType, typename ADataType, size_t ARank>
    requires CoreRankTensor<AType<ADataType, ARank>, 2, ADataType>
auto norm(Norm norm_type, const AType<ADataType, ARank> &a) -> RemoveComplexT<ADataType> {
    LabeledSection0();

    if (norm_type != Norm::Infinity) {
        return blas::lange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), nullptr);
    } else {
        std::vector<RemoveComplexT<ADataType>> work(a->dim(0), 0.0);
        return blas::lange(norm_type, a->dim(0), a->dim(1), a->data(), a->stride(0), work.data());
    }
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

    eint m   = A.dim(0);
    eint n   = A.dim(1);
    eint lda = A.stride(0);

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
    eint const info1 = blas::geqrf(n, k + 5, Y.data(), k + 5, tau.data());
    // Extract Matrix Q out of QR factorization
    eint const info2 = blas::orgqr(n, k + 5, tau.dim(0), Y.data(), k + 5, const_cast<const double *>(tau.data()));

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
    Tensor<T, 2>      R = A; // R is a copy of A
    Tensor<T, 2>      wr("Schur Real Buffer", n, n);
    Tensor<T, 2>      wi("Schur Imaginary Buffer", n, n);
    Tensor<T, 2>      U("Lyapunov U", n, n);
    std::vector<eint> sdim(1);
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
    Tensor<T, 2> A = _A;
    const eint   m = A.dim(0);
    const eint   n = A.dim(1);

    Tensor<double, 1> tau("tau", std::min(m, n));
    // Compute QR factorization of Y
    eint info = blas::geqrf(m, n, A.data(), n, tau.data());

    if (info != 0) {
        println_abort("{} parameter to geqrf has an illegal value.", -info);
    }

    // Extract Matrix Q out of QR factorization
    // eint info2 = blas::orgqr(m, n, tau.dim(0), A.data(), n, const_cast<const double *>(tau.data()));
    return {A, tau};
}

template <typename T>
auto q(const Tensor<T, 2> &qr, const Tensor<T, 1> &tau) -> Tensor<T, 2> {
    const eint m = qr.dim(1);
    const eint p = qr.dim(0);

    Tensor<T, 2> Q = qr;

    eint info = blas::orgqr(m, m, p, Q.data(), m, tau.data());
    if (info != 0) {
        println_abort("{} parameter to orgqr has an illegal value. {} {} {}", -info, m, m, p);
    }

    return Q;
}

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)
