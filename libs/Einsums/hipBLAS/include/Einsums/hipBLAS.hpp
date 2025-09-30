//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/hipBLASVendor/Vendor.hpp>

#include <type_traits>

namespace einsums::blas::gpu {

/**
 * @brief Perform a General Matrix Multiply (GEMM) operation.
 *
 * This function computes the product of two matrices,
 * \f[
 * \mathbf{C} := \alpha \mathbf{A}\mathbf{B} + \beta\mathbf{C}
 * \f]
 * where @f$\mathbf{A}@f$, @f$\mathbf{B}@f$, and @f$\mathbf{C}@f$ are matrices, and
 * @f$\alpha@f$ and @f$\beta@f$ are scalar values.
 *
 * @tparam T The datatype of the GEMM.
 * @param[in] transa Whether to transpose matrix a :
 *   - 'N' or 'n' for no transpose,
 *   - 'T' or 't' for transpose,
 *   - 'C' or 'c' for conjugate transpose.
 * @param[in] transb Whether to transpose matrix b .
 * @param[in] m The number of rows in matrix A and C.
 * @param[in] n The number of columns in matrix B and C.
 * @param[in] k The number of columns in matrix A and rows in matrix B.
 * @param[in] alpha The scalar alpha.
 * @param[in] a A pointer to the matrix A with dimensions `(lda, k)` when transa is 'N' or 'n', and `(lda, m)`
 * otherwise.
 * @param[in] lda Leading dimension of A, specifying the distance between two consecutive columns.
 * @param[in] b A pointer to the matrix B with dimensions `(ldb, n)` when transB is 'N' or 'n', and `(ldb, k)`
 * otherwise.
 * @param[in] ldb Leading dimension of B, specifying the distance between two consecutive columns.
 * @param[in] beta The scalar beta.
 * @param[inout] c A pointer to the matrix C with dimensions `(ldc, n)`.
 * @param[in] ldc Leading dimension of C, specifying the distance between two consecutive columns.
 *
 * @note The function performs one of the following matrix operations:
 * - If transA is 'N' or 'n' and transB is 'N' or 'n': \f$\mathbf{C} = \alpha\mathbf{AB} + \beta\mathbf{C}\f$
 * - If transA is 'N' or 'n' and transB is 'T' or 't': \f$\mathbf{C} = \alpha\mathbf{A}\mathbf{B}^T + \beta\mathbf{C}\f$
 * - If transA is 'T' or 't' and transB is 'N' or 'n': \f$\mathbf{C} = \alpha\mathbf{A}^T\mathbf{B} + \beta\mathbf{C}\f$
 * - If transA is 'T' or 't' and transB is 'T' or 't': \f$\mathbf{C} = \alpha\mathbf{A}^T\mathbf{B}^T + \beta\mathbf{C}\f$
 * - If transA is 'C' or 'c' and transB is 'N' or 'n': \f$\mathbf{C} = \alpha\mathbf{A}^H\mathbf{B} + \beta\mathbf{C}\f$
 * - If transA is 'C' or 'c' and transB is 'T' or 't': \f$\mathbf{C} = \alpha\mathbf{A}^H\mathbf{B}^Y + \beta\mathbf{C}\f$
 * - etc.
 *
 * @throws std::invalid_argument If @p transA or @p transB are invalid.
 * @throws std::domain_error If the values of @p m , @p n , or @p k are negative, or the values of @p lda , @p ldb , or @p ldc are
 * invalid.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void gemm(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c,
          int_t ldc);

#if !defined(DOXYGEN)
// These are the template specialization for the data types we support. If an unsupported data type
// is attempted a compiler error will occur.
template <>
inline void gemm<float>(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                        int_t ldb, float beta, float *c, int_t ldc) {
    blas::hip::sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<double>(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                         int_t ldb, double beta, double *c, int_t ldc) {
    blas::hip::dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<float>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha,
                                      std::complex<float> const *a, int_t lda, std::complex<float> const *b, int_t ldb,
                                      std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    blas::hip::cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
inline void gemm<std::complex<double>>(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha,
                                       std::complex<double> const *a, int_t lda, std::complex<double> const *b, int_t ldb,
                                       std::complex<double> beta, std::complex<double> *c, int_t ldc) {
    blas::hip::zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif

/**
 * @brief Computes a matrix-vector product using a general matrix.
 *
 * The gemv routine performs a matrix-vector operation defined as:
 * @f[
 * \mathbf{y} := \alpha \mathbf{A} \mathbf{x} + \beta \mathbf{y}
 * @f]
 * or
 * @f[
 * \mathbf{y} := \alpha \mathbf{A}^T \mathbf{x} + \beta \mathbf{y}
 * @f]
 * or
 * @f[
 * \mathbf{y} := \alpha \mathbf{A}^H \mathbf{x} + \beta \mathbf{y}
 * @f]
 *
 * @tparam T the underlying data type of the matrix and vector
 * @param[in] transa what to do with \p a - no trans, trans, conjg
 * @param[in] m specifies the number of rows of \p a
 * @param[in] n specifies the number of columns of \p a
 * @param[in] alpha Specifies the scaler alpha
 * @param[in] a Array, size lda * m
 * @param[in] lda Specifies the leading dimension of \p a as declared in the calling function
 * @param[in] x array, vector x
 * @param[in] incx Specifies the increment for the elements of \p x
 * @param[in] beta Specifies the scalar beta. When beta is set to zero, then \p y need not be set on input.
 * @param[inout] y array, vector y
 * @param[in] incy Specifies the increment for the elements of \p y .
 *
 * @throws std::invalid_argument If @p transA is invalid.
 * @throws std::domain_error If the values of @p m or @p n are negative, the value of @p lda is invalid, or either @p incx or @p incy is
 * zero.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void gemv(char transa, int_t m, int_t n, T alpha, T const *a, int_t lda, T const *x, int_t incx, T beta, T *y, int_t incy);

#if !defined(DOXYGEN)
template <>
inline void gemv<float>(char transa, int_t m, int_t n, float alpha, float const *a, int_t lda, float const *x, int_t incx, float beta,
                        float *y, int_t incy) {
    blas::hip::sgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<double>(char transa, int_t m, int_t n, double alpha, double const *a, int_t lda, double const *x, int_t incx, double beta,
                         double *y, int_t incy) {
    blas::hip::dgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<float>>(char transa, int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *a, int_t lda,
                                      std::complex<float> const *x, int_t incx, std::complex<float> beta, std::complex<float> *y,
                                      int_t incy) {
    blas::hip::cgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
inline void gemv<std::complex<double>>(char transa, int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *a, int_t lda,
                                       std::complex<double> const *x, int_t incx, std::complex<double> beta, std::complex<double> *y,
                                       int_t incy) {
    blas::hip::zgemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif

/**
 * @brief Performs diagonalization of a symmetrix matrix.
 *
 * The syev routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^T
 * @f]
 * In the above equation, @f$ \mathbf{A} @f$ is a real symmetric matrix, @f$ \mathbf{P} @f$ is a real orthogonal matrix whose columns are
 * the eigenvectors of @f$ \mathbf{A} @f$, and @f$ \mathbf{\Lambda} @f$ is a diagonal matrix, whose elements are the eigenvalues of @f$
 * \mathbf{A} @f$. The eigenvalues are stored in a vector form on exit.
 *
 * @tparam T The type the function will handle.
 * @param[in] job Whether to compute the eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] uplo Whether the matrix data is stored in the upper or lower triangle. Can be either 'u' or 'l', case insensitive.
 * @param[in] n The number of rows/columns of the input matrix.
 * @param[inout] a The input matrix. On output, it will be changed. If the eigenvectors are requested, then they will be placed
 * in the columns of @p a on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] w The output vector for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then no operations are performed and the first value in the work array is the
 * optimal work buffer size.
 * @param[in] lwork The size of the work array. If @p lwork is -1, then a workspace query is assumed. No operations will be performed
 * and the optimal workspace size will be put into the first element of @p work.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto syev(char job, char uplo, int_t n, T *a, int_t lda, T *w, T *work, int_t lwork) -> int_t;

#ifndef DOXYGEN
template <>
inline auto syev<float>(char job, char uplo, int_t n, float *a, int_t lda, float *w, float *work, int_t lwork) -> int_t {
    return blas::hip::ssyev(job, uplo, n, a, lda, w, work, lwork);
}

template <>
inline auto syev<double>(char job, char uplo, int_t n, double *a, int_t lda, double *w, double *work, int_t lwork) -> int_t {
    return blas::hip::dsyev(job, uplo, n, a, lda, w, work, lwork);
}
#endif

// No sterf in hipSolver.
#if 0
/**
 * @brief Computes the eigenvalues of a symmetric tridiagonal matrix.
 * The sterf routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^T
 * @f]
 * In the above equation, @f$ \mathbf{A} @f$ is a real symmetric tridiagonal matrix, @f$ \mathbf{P} @f$ is a real orthogonal matrix whose
 * columns are the eigenvectors of @f$ \mathbf{A} @f$, and @f$ \mathbf{\Lambda} @f$ is a diagonal matrix, whose elements are the eigenvalues
 * of @f$ \mathbf{A} @f$. The eigenvalues are stored in a vector form on exit.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements along the diagonal.
 * @param[inout] d The diagonal elements. On exit, it contains the eigenvalues.
 * @param[inout] e The off-diagonal elements. There is one fewer of these than the diagonal elements.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto sterf(int_t n, T *d, T *e) -> int_t;

#    ifndef DOXYGEN
template <>
inline auto sterf<float>(int_t n, float *d, float *e) -> int_t {
    return blas::hip::ssterf(n, d, e);
}

template <>
inline auto sterf<double>(int_t n, double *d, double *e) -> int_t {
    return blas::hip::dsterf(n, d, e);
}
#    endif
#endif

// No geev in hipSolver (no clue why).
#if 0
// Complex version
/**
 * @brief Performs diagonalization of a matrix.
 *
 * The syev routine finds the matrices that satisfy the following equations.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^{-1}
 * \mathbf{A}^T = \mathbf{L} \mathbf{\Lambda} \mathbf{L}^{-1}
 * @f]
 * In the above equation, @f$\mathbf{A}@f$ is a matrix, @f$\mathbf{P}@f$ is a complex-valued matrix whose columns are
 * the right eigenvectors of @f$\mathbf{A}@f$, @f$\mathbf{L}@f$ is a complex-valued matrix whose columns are the left eigenvectors of
 * @f$\mathbf{A}@f$, and @f$\mathbf{\Lambda}@f$ is a complex-valued diagonal matrix, whose elements are the eigenvalues of @f$ A @f$. The
 * eigenvalues are stored in a vector form on exit. The eigenvectors are stored in a special way if the input is a real matrix. If the input
 * is a complex matrix, then the eigenvectors are stored plainly in the columns of the appropriate output matrices.
 *
 * @tparam T The type this function handles.
 * @param[in] jobvl Whether to compute the left eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] jobvr Whether to compute the right eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] n The number of rows/columns of the input matrix.
 * @param[inout] a The input matrix. On output, it will be changed.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] w The output vector for the eigenvalues.
 * @param[out] vl The left eigenvector output. If @p jobvl is 'n', then this is not referenced and may be null.
 * @param[in] ldvl The leading dimension of the left eigenvectors. Even if not referenced, it must be at least 1.
 * @param[out] vr The right eigenvector output. If @p jobvr is 'n', then this is not referenced and may be null.
 * @param[in] ldvr The leading dimension of the right eigenvectors. Even if not referenced, it must be at least 1.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto geev(char jobvl, char jobvr, int_t n, T *a, int_t lda, AddComplexT<T> *w, T *vl, int_t ldvl, T *vr, int_t ldvr) -> int_t;

#    ifndef DOXYGEN
template <>
inline auto geev<float>(char jobvl, char jobvr, int_t n, float *a, int_t lda, std::complex<float> *w, float *vl, int_t ldvl, float *vr,
                        int_t ldvr) -> int_t {
    return blas::hip::sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<double>(char jobvl, char jobvr, int_t n, double *a, int_t lda, std::complex<double> *w, double *vl, int_t ldvl, double *vr,
                         int_t ldvr) -> int_t {
    return blas::hip::dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<float>>(char jobvl, char jobvr, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *w,
                                      std::complex<float> *vl, int_t ldvl, std::complex<float> *vr, int_t ldvr) -> int_t {
    return blas::hip::cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

template <>
inline auto geev<std::complex<double>>(char jobvl, char jobvr, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *w,
                                       std::complex<double> *vl, int_t ldvl, std::complex<double> *vr, int_t ldvr) -> int_t {
    return blas::hip::zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}
#    endif
#endif

/**
 * @brief Performs diagonalization of a Hermitian matrix.
 *
 * The heev routine finds the matrices that satisfy the following equation.
 * @f[
 * \mathbf{A} = \mathbf{P} \mathbf{\Lambda} \mathbf{P}^H
 * @f]
 * In the above equation, @f$\mathbf{A}@f$ is a Hermitian matrix, @f$\mathbf{P}@f$ is a unitary matrix whose columns are
 * the eigenvectors of @f$\mathbf{A}@f$, and @f$\mathbf{\Lambda}@f$ is a diagonal matrix, whose elements are the eigenvalues of
 * @f$\mathbf{A}@f$. The eigenvalues are stored in a vector form on exit.
 *
 * @tparam T The type this function handles.
 * @param[in] job Whether to compute the eigenvectors. Can be either 'n' or 'v', case insensitive.
 * @param[in] uplo Whether the matrix data is stored in the upper or lower triangle. Can be either 'u' or 'l', case insensitive.
 * @param[in] n The number of rows/columns of the input matrix.
 * @param[inout] a The input matrix. On output, it will be changed. If the eigenvectors are requested, then they will be placed
 * in the columns of @p a on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] w The output vector for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then no operations are performed and the first value in the work array is the
 * optimal work buffer size.
 * @param[in] lwork The size of the work array. If @p lwork is -1, then a workspace query is assumed. No operations will be performed
 * and the optimal workspace size will be put into the first element of @p work.
 *
 * @return 0 on success. If positive, this means that the algorithm did not converge. The return value indicates the number of eigenvalues
 * that were able to be computed. If negative, this means that one of the parameters was invalid. The absolute value indicates which
 * parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto heev(char job, char uplo, int_t n, std::complex<T> *a, int_t lda, T *w, std::complex<T> *work, int_t lwork) -> int_t;

#ifndef DOXYGEN
template <>
inline auto heev<float>(char job, char uplo, int_t n, std::complex<float> *a, int_t lda, float *w, std::complex<float> *work, int_t lwork)
    -> int_t {
    return blas::hip::cheev(job, uplo, n, a, lda, w, work, lwork);
}

template <>
inline auto heev<double>(char job, char uplo, int_t n, std::complex<double> *a, int_t lda, double *w, std::complex<double> *work,
                         int_t lwork) -> int_t {
    return blas::hip::zheev(job, uplo, n, a, lda, w, work, lwork);
}
#endif

/**
 * @brief Solve a system of linear equations.
 *
 * Solves equations of the following form.
 * @f[
 * \mathbf{A}\mathbf{x} = \mathbf{B}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of rows and columns of @f$\mathbf{A}@f$ and rows @f$\mathbf{B}@f$.
 * @param[in] nrhs The number of columns of @f$\mathbf{B}@f$
 * @param[inout] a The coefficient matrix. On exit, it contains the LU decomposition of @p a, where the lower-triangle matrix has unit
 * diagonal entries, which are not stored.
 * @param[in] lda The leading dimension of @p a.
 * @param[out] ipiv A list of pivots used in the decomposition.
 * @param[in] b The results matrix. Unlike normal LAPACK, this is not overwritten by the function.
 * @param[in] ldb The leading dimension of @p b.
 * @param[out] x The output matrix. On AMD cards, this is allowed --- even recommended --- to be the same as @c b . On NVidia cards, it is
 * not.
 * @param[in] ldx The leading dimension of the output matrix.
 *
 * @return 0 on success. If positive, then the matrix was singular. If negative, then a bad value was passed to the function.
 * The absolute value indicates which parameter was bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto gesv(int_t n, int_t nrhs, T *a, int_t lda, int_t *ipiv, T *b, int_t ldb, T *x, int_t ldx) -> int_t;

#ifndef DOXYGEN
template <>
inline auto gesv<float>(int_t n, int_t nrhs, float *a, int_t lda, int_t *ipiv, float *b, int_t ldb, float *x, int_t ldx) -> int_t {
    return blas::hip::sgesv(n, nrhs, a, lda, ipiv, b, ldb, x, ldx);
}

template <>
inline auto gesv<double>(int_t n, int_t nrhs, double *a, int_t lda, int_t *ipiv, double *b, int_t ldb, double *x, int_t ldx) -> int_t {
    return blas::hip::dgesv(n, nrhs, a, lda, ipiv, b, ldb, x, ldx);
}

template <>
inline auto gesv<std::complex<float>>(int_t n, int_t nrhs, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *b,
                                      int_t ldb, std::complex<float> *x, int_t ldx) -> int_t {
    return blas::hip::cgesv(n, nrhs, a, lda, ipiv, b, ldb, x, ldx);
}

template <>
inline auto gesv<std::complex<double>>(int_t n, int_t nrhs, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *b,
                                       int_t ldb, std::complex<double> *x, int_t ldx) -> int_t {
    return blas::hip::zgesv(n, nrhs, a, lda, ipiv, b, ldb, x, ldx);
}
#endif

/**
 * @brief Scales a vector by a value.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements to scale the vector by.
 * @param[in] alpha The scale factor.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements of the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void scal(int_t n, T const alpha, T *vec, int_t inc);

/**
 * @brief Scales a complex vector by a real value.
 *
 * @param[in] n The number of elements to scale the vector by.
 * @param[in] alpha The scale factor.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements of the vector.
 *
 * @versionadded{2.0.0}
 */
template <Complex T>
void scal(int_t n, RemoveComplexT<T> const alpha, T *vec, int_t inc);

#ifndef DOXYGEN
template <>
inline void scal<float>(int_t n, float const alpha, float *vec, int_t inc) {
    blas::hip::sscal(n, alpha, vec, inc);
}

template <>
inline void scal<double>(int_t n, double const alpha, double *vec, int_t inc) {
    blas::hip::dscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, std::complex<float> const alpha, std::complex<float> *vec, int_t inc) {
    blas::hip::cscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, std::complex<double> const alpha, std::complex<double> *vec, int_t inc) {
    blas::hip::zscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<float>>(int_t n, float const alpha, std::complex<float> *vec, int_t inc) {
    blas::hip::csscal(n, alpha, vec, inc);
}

template <>
inline void scal<std::complex<double>>(int_t n, double const alpha, std::complex<double> *vec, int_t inc) {
    blas::hip::zdscal(n, alpha, vec, inc);
}
#endif

/**
 * @brief Scales a vector by the reciprocal of a value.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The value to divide all the elements in the vector by.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements in the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void rscl(int_t n, T const alpha, T *vec, int_t inc);

/**
 * @brief Scales a complex vector by the reciprocal of a real value.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The value to divide all the elements in the vector by.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The spacing between elements in the vector.
 *
 * @versionadded{2.0.0}
 */
template <Complex T>
void rscl(int_t n, RemoveComplexT<T> const alpha, T *vec, int_t inc);

#ifndef DOXYGEN
template <>
inline void rscl<float>(int_t n, float const alpha, float *vec, int_t inc) {
    blas::hip::srscl(n, alpha, vec, inc);
}

template <>
inline void rscl<double>(int_t n, double const alpha, double *vec, int_t inc) {
    blas::hip::drscl(n, alpha, vec, inc);
}

template <>
inline void rscl<std::complex<float>>(int_t n, std::complex<float> const alpha, std::complex<float> *vec, int_t inc) {
    blas::hip::cscal(n, std::complex<float>{1.0} / alpha, vec, inc);
}

template <>
inline void rscl<std::complex<double>>(int_t n, std::complex<double> const alpha, std::complex<double> *vec, int_t inc) {
    blas::hip::zscal(n, std::complex<double>{1.0} / alpha, vec, inc);
}

template <>
inline void rscl<std::complex<float>>(int_t n, float const alpha, std::complex<float> *vec, int_t inc) {
    blas::hip::csrscl(n, alpha, vec, inc);
}

template <>
inline void rscl<std::complex<double>>(int_t n, double const alpha, std::complex<double> *vec, int_t inc) {
    blas::hip::zdrscl(n, alpha, vec, inc);
}
#endif

/**
 * Computes the dot product of two vectors. For complex vectors it is the non-conjugated dot product;
 * (c|z)dotu in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param[in] n length of the vectors
 * @param[in] x first vector
 * @param[in] incx how many elements to skip in x
 * @param[in] y second vector
 * @param[in] incy how many elements to skip in yo
 * @return result of the dot product
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto dot(int_t n, T const *x, int_t incx, T const *y, int_t incy) -> T;

#ifndef DOXYGEN
template <>
inline auto dot<float>(int_t n, float const *x, int_t incx, float const *y, int_t incy) -> float {
    return blas::hip::sdot(n, x, incx, y, incy);
}

template <>
inline auto dot<double>(int_t n, double const *x, int_t incx, double const *y, int_t incy) -> double {
    return blas::hip::ddot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float> {
    return blas::hip::cdot(n, x, incx, y, incy);
}

template <>
inline auto dot<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double> {
    return blas::hip::zdot(n, x, incx, y, incy);
}
#endif

/**
 * Computes the dot product of two vectors. For complex vector it is the conjugated dot product;
 * (c|z)dotc in BLAS nomenclature.
 *
 * @tparam T underlying data type
 * @param[in] n length of the vectors
 * @param[in] x first vector
 * @param[in] incx how many elements to skip in x
 * @param[in] y second vector
 * @param[in] incy how many elements to skip in yo
 * @return result of the dot product
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto dotc(int_t n, T const *x, int_t incx, T const *y, int_t incy) -> T;

#ifndef DOXYGEN
template <>
inline auto dotc<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, std::complex<float> const *y, int_t incy)
    -> std::complex<float> {
    return blas::hip::cdotc(n, x, incx, y, incy);
}

template <>
inline auto dotc<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, std::complex<double> const *y, int_t incy)
    -> std::complex<double> {
    return blas::hip::zdotc(n, x, incx, y, incy);
}
#endif

/**
 * @brief Adds two vectors together with a scale factor.
 *
 * Computes the following.
 * @f[
 * \mathbf{y} := \alpha\mathbf{x} + \mathbf{y}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha_x The scale factor for the input vector.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the output vector. It can be negative to go in reverse, or zero to broadcast values to @p y.
 * @param[inout] y The output vector.
 * @param[in] inc_y The skip value for the output vector. It can be negative to go in reverse, or zero to sum over the elements of @p x .
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void axpy(int_t n, T alpha_x, T const *x, int_t inc_x, T *y, int_t inc_y);

#ifndef DOXYGEN
template <>
inline void axpy<float>(int_t n, float alpha_x, float const *x, int_t inc_x, float *y, int_t inc_y) {
    blas::hip::saxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<double>(int_t n, double alpha_x, double const *x, int_t inc_x, double *y, int_t inc_y) {
    blas::hip::daxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<float>>(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x,
                                      std::complex<float> *y, int_t inc_y) {
    blas::hip::caxpy(n, alpha_x, x, inc_x, y, inc_y);
}

template <>
inline void axpy<std::complex<double>>(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x,
                                       std::complex<double> *y, int_t inc_y) {
    blas::hip::zaxpy(n, alpha_x, x, inc_x, y, inc_y);
}
#endif

/**
 * @brief Adds two vectors together with a scale factor.
 *
 * Computes the following.
 * @f[
 * \mathbf{y} := \alpha\mathbf{x} + \beta\mathbf{y}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha_x The scale factor for the input vector.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the output vector. It can be negative to go in reverse, or zero to broadcast values to @p y.
 * @param[in] b The scale factor for the output vector.
 * @param[inout] y The output vector.
 * @param[in] inc_y The skip value for the output vector. It can be negative to go in reverse, or zero to sum over the elements of @p x .
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void axpby(int_t n, T alpha_x, T const *x, int_t inc_x, T b, T *y, int_t inc_y);

#ifndef DOXYGEN
template <>
inline void axpby<float>(int_t n, float alpha_x, float const *x, int_t inc_x, float b, float *y, int_t inc_y) {
    blas::hip::saxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<double>(int_t n, double alpha_x, double const *x, int_t inc_x, double b, double *y, int_t inc_y) {
    blas::hip::daxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<float>>(int_t n, std::complex<float> alpha_x, std::complex<float> const *x, int_t inc_x,
                                       std::complex<float> b, std::complex<float> *y, int_t inc_y) {
    blas::hip::caxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}

template <>
inline void axpby<std::complex<double>>(int_t n, std::complex<double> alpha_x, std::complex<double> const *x, int_t inc_x,
                                        std::complex<double> b, std::complex<double> *y, int_t inc_y) {
    blas::hip::zaxpby(n, alpha_x, x, inc_x, b, y, inc_y);
}
#endif

/**
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{x}\mathbf{y}^T + \mathbf{A}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of entries in @p x.
 * @param[in] n The number of entries in @p y.
 * @param[in] alpha The scale factor for the outer product.
 * @param[in] x The left input vector.
 * @param[in] inc_x The skip value for the left input. May be negative to go in reverse.
 * @param[in] y The right input vector.
 * @param[in] inc_y The skip value for the right input. May be negative to go in reverse.
 * @param[inout] a The output matrix.
 * @param[in] lda The leading dimension of @p a.
 *
 * @throws std::domain_error If either of the dimension parameters are negative or the leading dimension of the matrix is less than the
 * number of columns.
 * @throws std::invalid_argument If either of the vector increments are zero.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void ger(int_t m, int_t n, T alpha, T const *x, int_t inc_x, T const *y, int_t inc_y, T *a, int_t lda);

/**
 * Performs a rank-1 update of a general matrix.
 *
 * The ?gerc routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{x}\mathbf{y}^H + \mathbf{A}
 * @f]
 *
 * @tparam T The type this function handle.
 * @param[in] m The number of entries in @p x.
 * @param[in] n The number of entries in @p y.
 * @param[in] alpha The scale factor for the outer product.
 * @param[in] x The left input vector.
 * @param[in] inc_x The skip value for the left input. May be negative to go in reverse.
 * @param[in] y The right input vector.
 * @param[in] inc_y The skip value for the right input. May be negative to go in reverse.
 * @param[inout] a The output matrix.
 * @param[in] lda The leading dimension of @p a.
 *
 * @throws std::domain_error If either of the dimension parameters are negative or the leading dimension of the matrix is less than the
 * number of columns.
 * @throws std::invalid_argument If either of the vector increments are zero.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void gerc(int_t m, int_t n, T alpha, T const *x, int_t inc_x, T const *y, int_t inc_y, T *a, int_t lda);

#ifndef DOXYGEN
template <>
inline void ger<float>(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    blas::hip::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<double>(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    blas::hip::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                                     std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    blas::hip::cger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void ger<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                                      std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    blas::hip::zger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<float>(int_t m, int_t n, float alpha, float const *x, int_t inc_x, float const *y, int_t inc_y, float *a, int_t lda) {
    blas::hip::sger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<double>(int_t m, int_t n, double alpha, double const *x, int_t inc_x, double const *y, int_t inc_y, double *a, int_t lda) {
    blas::hip::dger(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<std::complex<float>>(int_t m, int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t inc_x,
                                      std::complex<float> const *y, int_t inc_y, std::complex<float> *a, int_t lda) {
    blas::hip::cgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}

template <>
inline void gerc<std::complex<double>>(int_t m, int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t inc_x,
                                       std::complex<double> const *y, int_t inc_y, std::complex<double> *a, int_t lda) {
    blas::hip::zgerc(m, n, alpha, x, inc_x, y, inc_y, a, lda);
}
#endif

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row int_terchanges.
 *
 * The factorization has the form
 * @f[
 *   \mathbf{A} = \mathbf{PLU}
 * @f]
 * where @f$\mathbf{P}@f$ is a permutation matrix, @f$\mathbf{L}@f$ is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and @f$\mathbf{U}@f$ is upper
 * triangular (upper trapezoidal if m < n).
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows in the input.
 * @param[in] n The number of columns in the input.
 * @param[inout] a The input matrix. On exit, it contains the upper and lower triangular matrices. The elemnts of the lower
 * triangular matrix are not stored since they are all 1.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] ipiv The list of pivots.
 *
 * @return 0 on success. If positive, the matrix is singular and the result should not be used for solving systems of equations.
 * The decomposition was performed, though. If negative, one of the inputs had an invalid value. The absolute value indicates
 * which input it is.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto getrf(int_t m, int_t n, T *a, int_t lda, int_t *ipiv) -> int_t;

#ifndef DOXYGEN
template <>
inline auto getrf<float>(int_t m, int_t n, float *a, int_t lda, int_t *ipiv) -> int_t {
    return blas::hip::sgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<double>(int_t m, int_t n, double *a, int_t lda, int_t *ipiv) -> int_t {
    return blas::hip::dgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, int_t *ipiv) -> int_t {
    return blas::hip::cgetrf(m, n, a, lda, ipiv);
}

template <>
inline auto getrf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, int_t *ipiv) -> int_t {
    return blas::hip::zgetrf(m, n, a, lda, ipiv);
}
#endif

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of rows and columns of the matrix.
 * @param[in] a The input matrix after being processed by getrf.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] ipiv The pivots from getrf.
 * @param[out] c The calculated inverse.
 * @param[in] ldc The leading dimension of the output matrix.
 *
 * @return 0 on success. If positive, the matrix is singular and an inverse could not be computed. If negative,
 * one of the inputs is invalid, and the absolute value indicates which input is bad.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto getri(int_t n, T *a, int_t lda, int_t *ipiv, T *c, int_t ldc) -> int_t;

#ifndef DOXYGEN
template <>
inline auto getri<float>(int_t n, float *a, int_t lda, int_t *ipiv, float *c, int_t ldc) -> int_t {
    return blas::hip::sgetri(n, a, lda, ipiv, c, ldc);
}

template <>
inline auto getri<double>(int_t n, double *a, int_t lda, int_t *ipiv, double *c, int_t ldc) -> int_t {
    return blas::hip::dgetri(n, a, lda, ipiv, c, ldc);
}

template <>
inline auto getri<std::complex<float>>(int_t n, std::complex<float> *a, int_t lda, int_t *ipiv, std::complex<float> *c, int_t ldc)
    -> int_t {
    return blas::hip::cgetri(n, a, lda, ipiv, c, ldc);
}

template <>
inline auto getri<std::complex<double>>(int_t n, std::complex<double> *a, int_t lda, int_t *ipiv, std::complex<double> *c, int_t ldc)
    -> int_t {
    return blas::hip::zgetri(n, a, lda, ipiv, c, ldc);
}
#endif

// No lange in hipSolver.
#if 0
/**
 * Computes various matrix norms. The available norms are the 1-norm, Frobenius norm, Max-abs norm, and the infinity norm.
 *
 * @tparam T The type this matrix handles.
 * @param[in] norm_type The norm to compute. It is case insensitive. For the 1-norm, it should be '1' or 'o'. For the Frobenius norm it
 * should be 'f' or 'e'. For the max-abs norm it should be 'm'. For the infinity norm, it should be 'i'.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 * @param[in] A The matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[inout] work A work array. Only needed for the infinity norm.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto lange(char norm_type, int_t m, int_t n, T const *A, int_t lda, RemoveComplexT<T> *work) -> RemoveComplexT<T>;

#    ifndef DOXYGEN
template <>
inline auto lange<float>(char norm_type, int_t m, int_t n, float const *A, int_t lda, float *work) -> float {
    return blas::hip::slange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<double>(char norm_type, int_t m, int_t n, double const *A, int_t lda, double *work) -> double {
    return blas::hip::dlange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<float>>(char norm_type, int_t m, int_t n, std::complex<float> const *A, int_t lda, float *work) -> float {
    return blas::hip::clange(norm_type, m, n, A, lda, work);
}

template <>
inline auto lange<std::complex<double>>(char norm_type, int_t m, int_t n, std::complex<double> const *A, int_t lda, double *work)
    -> double {
    return blas::hip::zlange(norm_type, m, n, A, lda, work);
}
#    endif
#endif

/**
 * Compute the sum of the squares of the input vector without roundoff error.
 * @f[
 * scale^2 sumsq := \left|\mathbf{x}\right|^2 + scale^2 sumsq
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the vector.
 * @param[inout] scale The scale value used to avoid overflow/underflow. It is also used as an input to continue a previous calculation.
 * @param[inout] sumsq The result of the operation, scaled to avoid overflow/underflow. It is also used as an input to continue a previous
 * calculation.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void lassq(int_t n, T const *x, int_t incx, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq);

#ifndef DOXYGEN
template <>
inline void lassq<float>(int_t n, float const *x, int_t incx, float *scale, float *sumsq) {
    blas::hip::slassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<double>(int_t n, double const *x, int_t incx, double *scale, double *sumsq) {
    blas::hip::dlassq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx, float *scale, float *sumsq) {
    blas::hip::classq(n, x, incx, scale, sumsq);
}

template <>
inline void lassq<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx, double *scale, double *sumsq) {
    blas::hip::zlassq(n, x, incx, scale, sumsq);
}
#endif

/**
 * Compute the Euclidean norm of a vector.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the vector.
 *
 * @return The Euclidean norm of the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> nrm2(int_t n, T const *x, int_t incx);

#ifndef DOXYGEN
template <>
inline float nrm2<float>(int_t n, float const *x, int_t incx) {
    return blas::hip::snrm2(n, x, incx);
}

template <>
inline double nrm2<double>(int_t n, double const *x, int_t incx) {
    return blas::hip::dnrm2(n, x, incx);
}

template <>
inline float nrm2<std::complex<float>>(int_t n, std::complex<float> const *x, int_t incx) {
    return blas::hip::scnrm2(n, x, incx);
}

template <>
inline double nrm2<std::complex<double>>(int_t n, std::complex<double> const *x, int_t incx) {
    return blas::hip::dznrm2(n, x, incx);
}
#endif

// No divide-and-conquer algorithm in hipSolver.
#if 0
/**
 * Performs singular value decomposition for a matrix using the divide and conquer algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\Sigma V}^T
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] jobz What computation to do. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns of the input matrix.
 * @param[inout] a The input matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values output.
 * @param[out] u The U matrix from the singular value decomposition.
 * @param[in] ldu The leading dimension of U.
 * @param[out] vt The transpose of the V matrix from the singular value decomposition.
 * @param[in] ldvt The leading dimension of the transpose of the V matrix.
 *
 * @return 0 on success. If positive, the algorithm did not converge. If -4, then the input matrix had a NaN entry. If negative otherwise,
 * then one of the parameters had a bad value. The absolute value of the return gives the parameter.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto gesdd(char jobz, int_t m, int_t n, T *a, int_t lda, RemoveComplexT<T> *s, T *u, int_t ldu, T *vt, int_t ldvt) -> int_t;

#    ifndef DOXYGEN
template <>
inline auto gesdd<float>(char jobz, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt) -> int_t {
    return blas::hip::sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<double>(char jobz, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt, int_t ldvt)
    -> int_t {
    return blas::hip::dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<float>>(char jobz, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s, std::complex<float> *u,
                                       int_t ldu, std::complex<float> *vt, int_t ldvt) -> int_t {
    return blas::hip::cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline auto gesdd<std::complex<double>>(char jobz, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s, std::complex<double> *u,
                                        int_t ldu, std::complex<double> *vt, int_t ldvt) -> int_t {
    return blas::hip::zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}
#    endif
#endif

/**
 * Performs singular value decomposition for a matrix using the QR algorithm.
 *
 * @f[
 * \mathbf{A} = \mathbf{U\Sigma V}^T
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] jobu Whether to compute the U matrix. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param[in] jobvt Whether to compute the transpose of the V matrix. Case insensitive. Can be 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns of the input matrix.
 * @param[inout] a The input matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values output.
 * @param[out] u The U matrix from the singular value decomposition.
 * @param[in] ldu The leading dimension of U.
 * @param[out] vt The transpose of the V matrix from the singular value decomposition.
 * @param[in] ldvt The leading dimension of the transpose of the V matrix.
 * @param[inout] superb Temporary storage area for intermediates in the computation.
 *
 * @return 0 on success. If positive, the algorithm did not converge. If negative,
 * then one of the parameters had a bad value. The absolute value of the return gives the parameter.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto gesvd(char jobu, char jobvt, int_t m, int_t n, T *a, int_t lda, RemoveComplexT<T> *s, T *u, int_t ldu, T *vt, int_t ldvt, T *superb);

#ifndef DOXYGEN
template <>
inline auto gesvd<float>(char jobu, char jobvt, int_t m, int_t n, float *a, int_t lda, float *s, float *u, int_t ldu, float *vt, int_t ldvt,
                         float *superb) {
    return blas::hip::sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<double>(char jobu, char jobvt, int_t m, int_t n, double *a, int_t lda, double *s, double *u, int_t ldu, double *vt,
                          int_t ldvt, double *superb) {
    return blas::hip::dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<std::complex<float>>(char jobu, char jobvt, int_t m, int_t n, std::complex<float> *a, int_t lda, float *s,
                                       std::complex<float> *u, int_t ldu, std::complex<float> *vt, int_t ldvt,
                                       std::complex<float> *superb) {
    return blas::hip::cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}

template <>
inline auto gesvd<std::complex<double>>(char jobu, char jobvt, int_t m, int_t n, std::complex<double> *a, int_t lda, double *s,
                                        std::complex<double> *u, int_t ldu, std::complex<double> *vt, int_t ldvt,
                                        std::complex<double> *superb) {
    return blas::hip::zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
}
#endif

// No Schur decomposition in hipSolver.
#if 0
/**
 * Computes the Schur decomposition of a matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] jobvs Whether to compute the unitary matrix for the decomposition.
 * @param[in] n The number of rows and columns of the input matrix.
 * @param[inout] a The iput matrix. On exit, it contains the pseudotriangular matrix from the decomposition.
 * @param[in] lda The leading dimension of A.
 * @param[in] sdim The number of selected eigenvalues.
 * @param[out] wr The real components of the eigenvalues.
 * @param[out] wi The imaginary components of the eigenvaules.
 * @param[out] vs The Schur vector matrix.
 * @param[in] ldvs The leading dimension of the Schur vector matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had a bad value. The absolute value tells you which parameter it was.
 * If positive and less than or equal to the number of rows in the matrix, the QR algorithm failed to converge. If one more than the
 * number of rows in the matrix, the eigenvalues could not be reordered for some reason, usually due to eigenvalues being too close.
 * If two more than the number of rows in the matrix, then roundoff changed some of the eigenvalues.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto gees(char jobvs, int_t n, T *a, int_t lda, int_t *sdim, T *wr, T *wi, T *vs, int_t ldvs) -> int_t;

#    ifndef DOXYGEN
template <>
inline auto gees<float>(char jobvs, int_t n, float *a, int_t lda, int_t *sdim, float *wr, float *wi, float *vs, int_t ldvs) -> int_t {
    return blas::hip::sgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}

template <>
inline auto gees<double>(char jobvs, int_t n, double *a, int_t lda, int_t *sdim, double *wr, double *wi, double *vs, int_t ldvs) -> int_t {
    return blas::hip::dgees(jobvs, n, a, lda, sdim, wr, wi, vs, ldvs);
}
#    endif

/**
 * Computes the Schur decomposition of a matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] jobvs Whether to compute the unitary matrix for the decomposition.
 * @param[in] n The number of rows and columns of the input matrix.
 * @param[inout] a The iput matrix. On exit, it contains the pseudotriangular matrix from the decomposition.
 * @param[in] lda The leading dimension of A.
 * @param[in] sdim The number of selected eigenvalues.
 * @param[out] w The  eigenvalues.
 * @param[out] vs The Schur vector matrix.
 * @param[in] ldvs The leading dimension of the Schur vector matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had a bad value. The absolute value tells you which parameter it was.
 * If positive and less than or equal to the number of rows in the matrix, the QR algorithm failed to converge. If one more than the
 * number of rows in the matrix, the eigenvalues could not be reordered for some reason, usually due to eigenvalues being too close.
 * If two more than the number of rows in the matrix, then roundoff changed some of the eigenvalues.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto gees(char jobvs, int_t n, T *a, int_t lda, int_t *sdim, T *w, T *vs, int_t ldvs) -> int_t;

#    ifndef DOXYGEN
template <>
inline auto gees<std::complex<float>>(char jobvs, int_t n, std::complex<float> *a, int_t lda, int_t *sdim, std::complex<float> *w,
                                      std::complex<float> *vs, int_t ldvs) -> int_t {
    return blas::hip::cgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}

template <>
inline auto gees<std::complex<double>>(char jobvs, int_t n, std::complex<double> *a, int_t lda, int_t *sdim, std::complex<double> *w,
                                       std::complex<double> *vs, int_t ldvs) -> int_t {
    return blas::hip::zgees(jobvs, n, a, lda, sdim, w, vs, ldvs);
}
#    endif
#endif

// No Sylvester solver in hipSolver.
#if 0
/**
 * Solves a Sylvester equation. These equations look like the following.
 * @f[
 *  \mathbf{A}\mathbf{X} \pm \mathbf{X}\mathbf{B} = \alpha\mathbf{C}
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] trana Whether to transpose the A matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] tranb Whether to transpose the B matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] isgn Whether the sign in the equation is positive or negative.
 * @param[in] m The number of rows in X.
 * @param[in] n The number of columns in X.
 * @param[in] a The A matrix in Schur canonical form.
 * @param[in] lda The leading dimension of A.
 * @param[in] b The B matrix in Schur canonical form.
 * @param[in] ldb The leading dimension of B.
 * @param[inout] c The right hand side matrix. On exit, it contains the value of the X matrix that satisfies the equation.
 * @param[in] ldc The leading dimension of the C matrix.
 * @param[out] scale The scale factor for the right hand side matrix.
 *
 * @return 0 on success. If 1, then some eigenvalues were close and needed to be perturbed. If negative, then one of the inputs
 * had a bad value, and the absolute value of the return gives the parameter.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto trsyl(char trana, char tranb, int_t isgn, int_t m, int_t n, T const *a, int_t lda, T const *b, int_t ldb, T *c, int_t ldc,
           RemoveComplexT<T> *scale) -> int_t;

#    ifndef DOXYGEN
template <>
inline auto trsyl<float>(char trana, char tranb, int_t isgn, int_t m, int_t n, float const *a, int_t lda, float const *b, int_t ldb,
                         float *c, int_t ldc, float *scale) -> int_t {
    return blas::hip::strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<double>(char trana, char tranb, int_t isgn, int_t m, int_t n, double const *a, int_t lda, double const *b, int_t ldb,
                          double *c, int_t ldc, double *scale) -> int_t {
    return blas::hip::dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<float>>(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<float> const *a, int_t lda,
                                       std::complex<float> const *b, int_t ldb, std::complex<float> *c, int_t ldc, float *scale) -> int_t {
    return blas::hip::ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}

template <>
inline auto trsyl<std::complex<double>>(char trana, char tranb, int_t isgn, int_t m, int_t n, std::complex<double> const *a, int_t lda,
                                        std::complex<double> const *b, int_t ldb, std::complex<double> *c, int_t ldc, double *scale)
    -> int_t {
    return blas::hip::ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, scale);
}
#    endif
#endif

/**
 * Set up for computing the QR decomposition of a matrix.
 *
 * @f[
 * \mathbf{A} = \mathbf{QR}
 * @f]
 *
 * Here, @f$\mathbf{Q}@f$ is an orthogonal matrix and @f$\mathbf{R}@f$ is an upper triangular matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The input matrix. On exit, contains the data needed to compute the Q and R matrices. The
 * entries on and above the diagonal are the entries of the R matrix. The rest is needed to find the Q matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] tau On exit, holds the Householder reflector parameters for computing the Q matrix.
 *
 * @return 0 on success. If negative, one of the inputs had a bad value, and the absolute value of the return
 * tells you which one it was.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto geqrf(int_t m, int_t n, T *a, int_t lda, T *tau) -> int_t;

#ifndef DOXYGEN
template <>
inline auto geqrf<float>(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    return blas::hip::sgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<double>(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    return blas::hip::dgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return blas::hip::cgeqrf(m, n, a, lda, tau);
}

template <>
inline auto geqrf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return blas::hip::zgeqrf(m, n, a, lda, tau);
}
#endif

/**
 * Extract the Q matrix after a call to geqrf.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used in the calculation.
 * @param[inout] a The input matrix after being processed by geqrf.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[in] tau The scales for the elementary reflectors from geqrf.
 *
 * @return 0 on success. If negative, then one of the inputs had an invalid value, and the absolute value indicates
 * which parameter it is.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto orgqr(int_t m, int_t n, int_t k, T *a, int_t lda, T *tau) -> int_t;

#ifndef DOXYGEN
template <>
inline auto orgqr<float>(int_t m, int_t n, int_t k, float *a, int_t lda, float *tau) -> int_t {
    return blas::hip::sorgqr(m, n, k, a, lda, tau);
}

template <>
inline auto orgqr<double>(int_t m, int_t n, int_t k, double *a, int_t lda, double *tau) -> int_t {
    return blas::hip::dorgqr(m, n, k, a, lda, tau);
}

template <typename T>
auto ungqr(int_t m, int_t n, int_t k, T *a, int_t lda, T *tau) -> int_t;

template <>
inline auto ungqr<std::complex<float>>(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return blas::hip::cungqr(m, n, k, a, lda, tau);
}

template <>
inline auto ungqr<std::complex<double>>(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return blas::hip::zungqr(m, n, k, a, lda, tau);
}
#endif

// No LQ decomposition in hipSolver.
#if 0
/**
 * Set up for computing the LQ decomposition of a matrix.
 *
 * @f[
 * \mathbf{A} = \mathbf{LQ}
 * @f]
 *
 * Here, @f$\mathbf{Q}@f$ is an orthogonal matrix and @f$\mathbf{L}@f$ is a lower triangular matrix.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The input matrix. On exit, contains the data needed to compute the Q and L matrices. The
 * entries on and below the diagonal are the entries of the L matrix. The rest is needed to find the Q matrix.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] tau On exit, holds the Householder reflector parameters for computing the Q matrix.
 *
 * @return 0 on success. If negative, one of the inputs had a bad value, and the absolute value of the return
 * tells you which one it was.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto gelqf(int_t m, int_t n, T *a, int_t lda, T *tau) -> int_t;

#ifndef DOXYGEN
template <>
inline auto gelqf<float>(int_t m, int_t n, float *a, int_t lda, float *tau) -> int_t {
    return blas::hip::sgelqf(m, n, a, lda, tau);
}

template <>
inline auto gelqf<double>(int_t m, int_t n, double *a, int_t lda, double *tau) -> int_t {
    return blas::hip::dgelqf(m, n, a, lda, tau);
}

template <>
inline auto gelqf<std::complex<float>>(int_t m, int_t n, std::complex<float> *a, int_t lda, std::complex<float> *tau) -> int_t {
    return blas::hip::cgelqf(m, n, a, lda, tau);
}

template <>
inline auto gelqf<std::complex<double>>(int_t m, int_t n, std::complex<double> *a, int_t lda, std::complex<double> *tau) -> int_t {
    return blas::hip::zgelqf(m, n, a, lda, tau);
}
#endif

/**
 * Extract the Q matrix after a call to gelqf.
 *
 * @tparam T The type this function handles.
 * @param[in] m The number of rows of the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used in the calculation.
 * @param[inout] a The input matrix after being processed by gelqf.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[in] tau The scales for the elementary reflectors from gelqf.
 *
 * @return 0 on success. If negative, then one of the inputs had an invalid value, and the absolute value indicates
 * which parameter it is.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
auto orglq(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

#ifndef DOXYGEN
template <>
inline auto orglq<float>(int_t m, int_t n, int_t k, float *a, int_t lda, float const *tau) -> int_t {
    return blas::hip::sorglq(m, n, k, a, lda, tau);
}

template <>
inline auto orglq<double>(int_t m, int_t n, int_t k, double *a, int_t lda, double const *tau) -> int_t {
    return blas::hip::dorglq(m, n, k, a, lda, tau);
}

template <typename T>
auto unglq(int_t m, int_t n, int_t k, T *a, int_t lda, T const *tau) -> int_t;

template <>
inline auto unglq<std::complex<float>>(int_t m, int_t n, int_t k, std::complex<float> *a, int_t lda, std::complex<float> const *tau)
    -> int_t {
    return blas::hip::cunglq(m, n, k, a, lda, tau);
}

template <>
inline auto unglq<std::complex<double>>(int_t m, int_t n, int_t k, std::complex<double> *a, int_t lda, std::complex<double> const *tau)
    -> int_t {
    return blas::hip::zunglq(m, n, k, a, lda, tau);
}
#endif
#endif

/**
 * Copy data from one vector to another.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements to copy.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the input vector. If negative, the vector is traversed backwards. If zero, the values are broadcast
 * to the output vector.
 * @param[out] y The output vector.
 * @param[in] inc_y The skip value for the output vector. If negative, the vector is traversed backwards.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void copy(int_t n, T const *x, int_t inc_x, T *y, int_t inc_y);

#ifndef DOXYGEN
template <>
inline void copy<float>(int_t n, float const *x, int_t inc_x, float *y, int_t inc_y) {
    blas::hip::scopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<double>(int_t n, double const *x, int_t inc_x, double *y, int_t inc_y) {
    blas::hip::dcopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<std::complex<float>>(int_t n, std::complex<float> const *x, int_t inc_x, std::complex<float> *y, int_t inc_y) {
    blas::hip::ccopy(n, x, inc_x, y, inc_y);
}

template <>
inline void copy<std::complex<double>>(int_t n, std::complex<double> const *x, int_t inc_x, std::complex<double> *y, int_t inc_y) {
    blas::hip::zcopy(n, x, inc_x, y, inc_y);
}
#endif

/**
 * Scales a general matrix. The scale factor is <tt> cto / cfrom </tt>, but the scale is performed without overflow/underflow.
 *
 * @tparam T The type this function handles.
 * @param[in] type The type of matrix. Case insensitive. 'g' is for general matrices, 'l' is for lower triangular matrices, 'u' if for upper
 * triangular matrices, 'h' is for hessenberg matrices, 'b' is for symmetric band matrices with lower bandwidth @p kl and upper bandwidth of
 * @p ku and with only the lower half stored, 'q' is the same as 'b' but with the upper half stored instead, and 'z' is the same as 'b' but
 * with a more complicated storage scheme.
 * @param[in] kl The lower bandwidth of the matrix. Only used if the type is 'b', 'q', or 'z'.
 * @param[in] ku The upper bandwidth of the matrix. Only used if the type is 'b', 'q', or 'z'.
 * @param[in] cfrom The denominator for the scale.
 * @param[in] cto The numerator for the scale.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 * @param[inout] A The matrix being scaled.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value of the return indicates which
 * parameter it was.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
int_t lascl(char type, int_t kl, int_t ku, T cfrom, T cto, int_t m, int_t n, T *A, int_t lda);

#ifndef DOXYGEN
template <>
inline int_t lascl<float>(char type, int_t kl, int_t ku, float cfrom, float cto, int_t m, int_t n, float *vec, int_t lda) {
    return blas::hip::slascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}

template <>
inline int_t lascl<double>(char type, int_t kl, int_t ku, double cfrom, double cto, int_t m, int_t n, double *vec, int_t lda) {
    return blas::hip::dlascl(type, kl, ku, cfrom, cto, m, n, vec, lda);
}
#endif

/**
 * Computes the direct product between two vectors.
 *
 * @f[
 * z_i := z_i + \alpha x_i y_i
 * @f]
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The first input vector.
 * @param[in] incx The skip value for the first vector.
 * @param[in] y The second input vector.
 * @param[in] incy The skip value for the second vector.
 * @param[inout] z The accumulation vector.
 * @param[in] incz The skip value for the accumulation vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void dirprod(int_t n, T alpha, T const *x, int_t incx, T const *y, int_t incy, T *z, int_t incz);

#ifndef DOXYGEN
template <>
inline void dirprod<float>(int_t n, float alpha, float const *x, int_t incx, float const *y, int_t incy, float *z, int_t incz) {
    blas::hip::sdirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<double>(int_t n, double alpha, double const *x, int_t incx, double const *y, int_t incy, double *z, int_t incz) {
    blas::hip::ddirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<std::complex<float>>(int_t n, std::complex<float> alpha, std::complex<float> const *x, int_t incx,
                                         std::complex<float> const *y, int_t incy, std::complex<float> *z, int_t incz) {
    blas::hip::cdirprod(n, alpha, x, incx, y, incy, z, incz);
}

template <>
inline void dirprod<std::complex<double>>(int_t n, std::complex<double> alpha, std::complex<double> const *x, int_t incx,
                                          std::complex<double> const *y, int_t incy, std::complex<double> *z, int_t incz) {
    blas::hip::zdirprod(n, alpha, x, incx, y, incy, z, incz);
}
#endif

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the absolute values of the components, not the magnitudes.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> asum(int_t n, T const *x, int_t incx);

#ifndef DOXYGEN
template <>
inline float asum(int_t n, float const *x, int_t incx) {
    return blas::hip::sasum(n, x, incx);
}

template <>
inline double asum(int_t n, double const *x, int_t incx) {
    return blas::hip::dasum(n, x, incx);
}

template <>
inline float asum(int_t n, std::complex<float> const *x, int_t incx) {
    return blas::hip::scasum(n, x, incx);
}

template <>
inline double asum(int_t n, std::complex<double> const *x, int_t incx) {
    return blas::hip::dzasum(n, x, incx);
}
#endif

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the magnitudes.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
RemoveComplexT<T> sum1(int_t n, T const *x, int_t incx);

#ifndef DOXYGEN
template <>
inline float sum1(int_t n, float const *x, int_t incx) {
    return blas::hip::sasum(n, x, incx);
}

template <>
inline double sum1(int_t n, double const *x, int_t incx) {
    return blas::hip::dasum(n, x, incx);
}

template <>
inline float sum1(int_t n, std::complex<float> const *x, int_t incx) {
    return blas::hip::scsum1(n, x, incx);
}

template <>
inline double sum1(int_t n, std::complex<double> const *x, int_t incx) {
    return blas::hip::dzsum1(n, x, incx);
}
#endif

/**
 * Take the conjugate of a vector. Does nothing if the vector is real.
 *
 * @tparam T The type this function handles.
 * @param[in] n The number of elements in the vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the vector.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void lacgv(int_t n, T *x, int_t incx);

#ifndef DOXYGEN
template <>
inline void lacgv<float>(int_t n, float *x, int_t incx) {
    // Conjugating real values does nothing.
}

template <>
inline void lacgv<double>(int_t n, double *x, int_t incx) {
    // Conjugating real values does nothing.
}

template <>
inline void lacgv<std::complex<float>>(int_t n, std::complex<float> *x, int_t incx) {
    blas::hip::clacgv(n, x, incx);
}

template <>
inline void lacgv<std::complex<double>>(int_t n, std::complex<double> *x, int_t incx) {
    blas::hip::zlacgv(n, x, incx);
}
#endif

} // namespace einsums::blas::gpu