//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"
#include "einsums/_Compiler.hpp"

#include "einsums/Blas.hpp"
#include "einsums/STL.hpp"
#include "einsums/Section.hpp"
#include "einsums/Tensor.hpp"
#include "einsums/Utilities.hpp"
#include "einsums/utility/ComplexTraits.hpp"
#include "einsums/utility/SmartPointerTraits.hpp"
#include "einsums/utility/TensorTraits.hpp"
#ifdef __HIP__
#    include "einsums/DeviceTensor.hpp"
#    include "einsums/linear_algebra_imp/GPULinearAlgebra.hpp"
#endif
#include "einsums/linear_algebra_imp/BaseLinearAlgebra.hpp"
#include "einsums/linear_algebra_imp/BlockLinearAlgebra.hpp"
#include "einsums/linear_algebra_imp/BlockTiledLinearAlgebra.hpp"
#include "einsums/linear_algebra_imp/TiledLinearAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <complex>

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
void sum_square(const AType<ADataType, ARank> &a, RemoveComplexT<ADataType> *scale, RemoveComplexT<ADataType> *sumsq) {
    LabeledSection0();
    detail::sum_square(a, scale, sumsq);
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
        requires InSamePlace<AType<T, Rank>, BType<T, Rank>, Rank, Rank, T, T>;
        requires InSamePlace<AType<T, Rank>, CType<T, Rank>, Rank, Rank, T, T>;
        requires std::convertible_to<U, T>;
    }
void gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, const U beta, CType<T, Rank> *C) {
    LabeledSection0();
    detail::gemm<TransA, TransB>(alpha, A, B, beta, C);
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
        requires InSamePlace<AType<T, Rank>, BType<T, Rank>, 2, 2, T, T>;
        requires std::is_same_v<remove_view_t<AType, Rank, T>, remove_view_t<BType, Rank, T>>;
        requires std::convertible_to<U, T>;
    }
auto gemm(const U alpha, const AType<T, Rank> &A, const BType<T, Rank> &B) -> remove_view_t<AType, Rank, T> {
    LabeledSection0();

    remove_view_t<AType, Rank, T> C{"gemm result", TransA ? A.dim(1) : A.dim(0), TransB ? B.dim(0) : B.dim(1)};
    gemm<TransA, TransB>(static_cast<T>(alpha), A, B, static_cast<T>(0.0), &C);

    return C;
}

/**
 * @brief Computes a common double multiplication between two matrices.
 *
 * Computes @f$ C = OP(B)^T OP(A) OP(B) @f$.
 */
template <bool TransA, bool TransB, template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, size_t Rank, typename T>
    requires requires {
        requires InSamePlace<AType<T, Rank>, BType<T, Rank>, 2, 2, T, T>;
        requires InSamePlace<AType<T, Rank>, CType<T, Rank>, 2, 2, T, T>;
    }
void symm_gemm(const AType<T, Rank> &A, const BType<T, Rank> &B, CType<T, Rank> *C) {
    LabeledSection0();

    detail::symm_gemm<TransA, TransB>(A, B, C);
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
        requires InSamePlace<AType<T, ARank>, XType<T, XYRank>, 2, 1, T, T>;
        requires InSamePlace<AType<T, ARank>, YType<T, XYRank>, 2, 1, T, T>;
        requires std::convertible_to<U, T>; // Make sure the alpha and beta can be converted to T
    }
void gemv(const U alpha, const AType<T, ARank> &A, const XType<T, XYRank> &z, const U beta, YType<T, XYRank> *y) {
    LabeledSection1(fmt::format("<TransA={}>", TransA));

    detail::gemv<TransA>(alpha, A, z, beta, y);
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
template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires InSamePlace<AType<T, ARank>, WType<T, WRank>, 2, 1, T, T>;
        requires !Complex<T>;
    }
void syev(AType<T, ARank> *A, WType<T, WRank> *W) {

    LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));
    detail::syev<ComputeEigenvectors>(A, W);
}

template <bool ComputeLeftRightEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <Complex, size_t> typename WType, size_t WRank, typename T>
    requires InSamePlace<AType<T, ARank>, WType<AddComplexT<T>, WRank>, 2, 1, T, AddComplexT<T>>
void geev(AType<T, ARank> *A, WType<AddComplexT<T>, WRank> *W, AType<T, ARank> *lvecs, AType<T, ARank> *rvecs) {
    LabeledSection1(fmt::format("<ComputeLeftRightEigenvectors={}>", ComputeLeftRightEigenvectors));

    detail::geev<ComputeLeftRightEigenvectors>(A, W, lvecs, rvecs);
}

template <bool ComputeEigenvectors = true, template <typename, size_t> typename AType, size_t ARank,
          template <typename, size_t> typename WType, size_t WRank, typename T>
    requires requires {
        requires InSamePlace<AType<T, ARank>, WType<RemoveComplexT<T>, WRank>, 2, 1, T, RemoveComplexT<T>>;
        requires Complex<T>;
    }
void heev(AType<T, ARank> *A, WType<RemoveComplexT<T>, WRank> *W) {
    LabeledSection1(fmt::format("<ComputeEigenvectors={}>", ComputeEigenvectors));
    detail::heev<ComputeEigenvectors>(A, W);
}

template <template <typename, size_t> typename AType, size_t ARank, template <typename, size_t> typename BType, size_t BRank, typename T>
    requires(InSamePlace<AType<T, ARank>, BType<T, BRank>, 2, 2, T, T>)
auto gesv(AType<T, ARank> *A, BType<T, BRank> *B) -> int {

    LabeledSection0();
    return detail::gesv(A, B);
}

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
 * @tparam T The underlying data type (required to be real)
 * @tparam ComputeEigenvectors If true, eigenvalues and eigenvectors are computed. If false, only eigenvalues are computed. Defaults to
 * true.
 * @param A The symmetric matrix A in the leading N-by-N upper triangular part of A.
 * @return std::tuple<Tensor<T, 2>, Tensor<T, 1>> The eigenvectors and eigenvalues.
 */
template <template <typename, size_t> typename AType, typename T, bool ComputeEigenvectors = true>
auto syev(const AType<T, 2> &A)
    -> std::tuple<remove_view_t<AType, 2, T>,
#ifdef __HIP__
                  std::conditional_t<einsums::detail::IsDeviceRankTensorV<AType<T, 1>, 1, T>, DeviceTensor<T, 1>, Tensor<T, 1>>
#else
                  Tensor<T, 1>
#endif
                  > {
    LabeledSection0();

    assert(A.dim(0) == A.dim(1));

    remove_view_t<AType, 2, T> a = A;
#ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceRankTensorV<AType<T, 2>, 2, T>) {
        DeviceTensor<T, 1> w{"eigenvalues", einsums::detail::DEV_ONLY, A.dim(0)};
        syev<ComputeEigenvectors>(&a, &w);
        return std::make_tuple(a, w);
    } else {
#endif
        Tensor<T, 1> w{"eigenvalues", A.dim(0)};

        syev<ComputeEigenvectors>(&a, &w);

        return std::make_tuple(a, w);
#ifdef __HIP__
    }
#endif
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
void scale(T scale, AType<T, ARank> *A) {
    LabeledSection0();

    detail::scale(scale, A);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires(ARank == 2)
void scale_row(size_t row, T scale, AType<T, ARank> *A) {
    LabeledSection0();

    detail::scale_row(row, scale, A);
}

template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires(ARank == 2)
void scale_column(size_t col, T scale, AType<T, ARank> *A) {
    LabeledSection0();

    detail::scale_column(col, scale, A);
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
 */
template <template <typename, size_t> typename AType, size_t ARank, typename T>
    requires(ARank == 2)
auto pow(const AType<T, ARank> &a, T alpha, T cutoff = std::numeric_limits<T>::epsilon()) -> remove_view_t<AType, 2, T> {
    LabeledSection0();

#ifdef __HIP__
    if constexpr (einsums::detail::IsDeviceRankTensorV<AType<T, ARank>, ARank, T>) {
        DeviceTensor<T, 1> Evals(Dim<1>{a.dim(0)}, ::einsums::detail::DEV_ONLY);

        remove_view_t<AType, 2, T> Evecs = create_tensor_like(a);

        remove_view_t<AType, 2, T> Diag = create_tensor_like(a);

        remove_view_t<AType, 2, T> out = create_tensor_like(a);
        remove_view_t<AType, 2, T> temp = create_tensor_like(a);

        Evecs.assign(a);

        syev<true>(&Evecs, &Evals);

        Diag.zero();

        detail::detail::gpu::eig_to_diag<<<dim3(32), dim3(1), 0, gpu::get_stream()>>>(Diag.data(), Diag.dim(0), Diag.stride(0),
                                                                                      Evals.data(), alpha);

        symm_gemm<false, false>(Diag, Evecs, &out);

        return out;
    } else {
#endif

        assert(a.dim(0) == a.dim(1));

        size_t                     n  = a.dim(0);
        remove_view_t<AType, 2, T> a1 = a;
        remove_view_t<AType, 2, T> result = create_tensor_like(a);
        result.set_name("pow result");
        Tensor<T, 1>               e{"e", n};
        result.zero();

        // Diagonalize
        syev<true>(&a1, &e);

        remove_view_t<AType, 2, T> a2(a1);

        // Determine the largest magnitude of the eigenvalues to use as a scaling factor for the cutoff.

        double max_e = 0;
        // Block tensors don't have sorted eigenvalues, so we can't make assumptions about ordering.
        for(int i = 0; i < n; i++) {
            if(std::fabs(e(i)) > max_e) {
                max_e = std::fabs(e(i));
            }
        }

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
#ifdef __HIP__
    }
#endif
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T>
    requires InSamePlace<AType<T, 1>, BType<T, 1>, 1, 1, T, T>
auto dot(const AType<T, 1> &A, const BType<T, 1> &B) -> T {
    LabeledSection0();

    return detail::dot(A, B);
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires InSamePlace<AType<T, Rank>, BType<T, Rank>, Rank, Rank, T, T>
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> T {

    LabeledSection0();

    return detail::dot(A, B);
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T>
    requires InSamePlace<AType<T, 1>, BType<T, 1>, 1, 1, T, T>
auto true_dot(const AType<T, 1> &A, const BType<T, 1> &B) -> RemoveComplexT<T> {
    LabeledSection0();

    return detail::true_dot(A, B);
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType, typename T, size_t Rank>
    requires InSamePlace<AType<T, Rank>, BType<T, Rank>, Rank, Rank, T, T>
auto true_dot(const AType<T, Rank> &A, const BType<T, Rank> &B) -> RemoveComplexT<T> {

    LabeledSection0();

    return detail::true_dot(A, B);
}

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
    requires requires {
        requires InSamePlace<AType<T, Rank>, BType<T, Rank>, Rank, Rank, T, T>;
        requires InSamePlace<AType<T, Rank>, CType<T, Rank>, Rank, Rank, T, T>;
    }
auto dot(const AType<T, Rank> &A, const BType<T, Rank> &B, const CType<T, Rank> &C) -> T {

    LabeledSection0();
    return detail::dot(A, B, C);
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires InSamePlace<XType<T, Rank>, YType<T, Rank>, Rank, Rank, T, T>
void axpy(T alpha, const XType<T, Rank> &X, YType<T, Rank> *Y) {
    LabeledSection0();

    detail::axpy(alpha, X, Y);
}

template <template <typename, size_t> typename XType, template <typename, size_t> typename YType, typename T, size_t Rank>
    requires InSamePlace<XType<T, Rank>, YType<T, Rank>, Rank, Rank, T, T>
void axpby(T alpha, const XType<T, Rank> &X, T beta, YType<T, Rank> *Y) {
    LabeledSection0();

    detail::axpby(alpha, X, beta, Y);
}

template <template <typename, size_t> typename XYType, size_t XYRank, template <typename, size_t> typename AType, typename T, size_t ARank>
    requires InSamePlace<AType<T, ARank>, XYType<T, XYRank>, 2, 1, T, T>
void ger(T alpha, const XYType<T, XYRank> &X, const XYType<T, XYRank> &Y, AType<T, ARank> *A) {
    LabeledSection0();

    detail::ger(alpha, X, Y, A);
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
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
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

template<template<typename, size_t> typename AType, typename ADataType, size_t ARank>
RemoveComplexT<ADataType> vec_norm(const AType<ADataType, ARank> &a) {
    return std::sqrt(std::abs(true_dot(a, a)));
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

template <template <typename, size_t> typename AType, template <typename, size_t> typename BType,
          template <typename, size_t> typename CType, typename T, size_t Rank>
void direct_product(T alpha, const AType<T, Rank> &A, const BType<T, Rank> &B, T beta, CType<T, Rank> *C) {
    LabeledSection0();

    detail::direct_product(alpha, A, B, beta, C);
}

END_EINSUMS_NAMESPACE_HPP(einsums::linear_algebra)
