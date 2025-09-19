//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/Types.hpp>
#include <Einsums/Config/ExportDefinitions.hpp>

#include <complex>

namespace einsums::blas::hip {

/**
 * Performs matrix multiplication for general square matrices.
 *
 * @param[in] transa Whether to transpose the left matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] transb Whether to transpose the right matrix. Case insensitive. Can be 'c', 't', or 'n'.
 * @param[in] m The number of rows in the output matrix.
 * @param[in] n The number of columns in the output matrix.
 * @param[in] k The number of columns in the first input matrix if @p transa is 'n'. May be different.
 * @param[in] alpha The scale factor for the product.
 * @param[in] a The left matrix.
 * @param[in] lda The leading dimension for the left matrix.
 * @param[in] b The right matrix.
 * @param[in] ldb The leading dimension for the right matrix.
 * @param[in] beta The scale factor for the output matrix.
 * @param[inout] c The output matrix.
 * @param[in] ldc The leading dimension for the output matrix.
 *
 * @throws std::invalid_argument If @p transa or @p transb are not one of the valid options.
 * @throws std::domain_error If any of the integer arguments have invalid values.
 *
 * @versionadded{2.0.0}
 */
void sgemm(char transa, char transb, int m, int n, int k, float alpha, float const *a, int lda, float const *b, int ldb, float beta,
           float *c, int ldc);
/// @copydoc sgemm
void dgemm(char transa, char transb, int m, int n, int k, double alpha, double const *a, int lda, double const *b, int ldb, double beta,
           double *c, int ldc);
/// @copydoc sgemm
void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, std::complex<float> const *a, int lda,
           std::complex<float> const *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc);
/// @copydoc sgemm
void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, std::complex<double> const *a, int lda,
           std::complex<double> const *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc);

/**
 * Performs matrix vector multiplication.
 *
 * @param[in] transa Whether to transpose the matrix. Case insensitive. Can be 'n', 'c', or 't'.
 * @param[in] m The number of entries in the input vector.
 * @param[in] n The number of entries in the output vector.
 * @param[in] alpha The scale factor for the multiplication.
 * @param[in] a The matrix.
 * @param[in] lda The leading dimension for the matrix.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the input vector. If it is negative, then the multiplication proceeds in reverse.
 * @param[in] beta The scale factor for the output vector.
 * @param[inout] y The output vector.
 * @param[in] incy The skip value for the output vector. If it is negative, then the multiplication proceeds in reverse.
 *
 * @throws std::invalid_argument If @p transa or @p transb are not one of the valid options.
 * @throws std::domain_error If any of the integer arguments have invalid values.
 *
 * @versionadded{2.0.0}
 */
void sgemv(char transa, int m, int n, float alpha, float const *a, int lda, float const *x, int incx, float beta, float *y, int incy);
/// @copydoc sgemv
void dgemv(char transa, int m, int n, double alpha, double const *a, int lda, double const *x, int incx, double beta, double *y, int incy);
/// @copydoc sgemv
void cgemv(char transa, int m, int n, std::complex<float> alpha, std::complex<float> const *a, int lda, std::complex<float> const *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy);
/// @copydoc sgemv
void zgemv(char transa, int m, int n, std::complex<double> alpha, std::complex<double> const *a, int lda, std::complex<double> const *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy);

/**
 * Scales and adds two vectors.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] a The scale factor for the first vector.
 * @param[in] x The input vector.
 * @param[in] incx The skip value for the input vector. If it is negative, then the vector is traversed backwards. If it is zero, then the
 * result will be broadcast to the output vector.
 * @param[in] b The scale factor for the output vector.
 * @param[inout] y The output vector.
 * @param[in] incy The skip value for the output vector. If it is negative, then the vector is traversed backwards.
 *
 * @versionadded{2.0.0}
 */
void saxpby(int const n, float const a, float const *x, int const incx, float const b, float *y, int const incy);
/// @copydoc saxpby
void daxpby(int const n, double const a, double const *x, int const incx, double const b, double *y, int const incy);
/// @copydoc saxpby
void caxpby(int const n, std::complex<float> const a, std::complex<float> const *x, int const incx, std::complex<float> const b,
            std::complex<float> *y, int const incy);
/// @copydoc saxpby
void zaxpby(int const n, std::complex<double> const a, std::complex<double> const *x, int const incx, std::complex<double> const b,
            std::complex<double> *y, int const incy);

/**
 * Performs symmetric matrix diagonalization.
 *
 * @param[in] job Whether to compute the eigenvectors. Case insensitive. Can be either 'n' or 'v'.
 * @param[in] uplo Whether the data is stored in the upper or lower triangle of the input. Case insensitive. Can be either 'u' or 'l'.
 * @param[in] n The number of rows and columns of the matrix.
 * @param[inout] a The matrix to factor. On exit, its data is overwritten. If the vectors are computed, they are placed in the columns of
 * the matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] w The output for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then the optimal work array size is placed in the first element of this array.
 * @param[in] lwork The size of the work array. If it is -1, then a workspace query is performed, and the optimal workspace size is put in
 * the first element of @p work.
 *
 * @return 0 on success. If positive, then the eigenvalue algorithm did not converge. If negative, then one of the parameters was given a
 * bad value. The absolute value indicates which parameter it was.
 *
 * @versionadded{2.0.0}
 */
auto ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) -> int;
/// @copydoc ssyev
auto dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) -> int;

/**
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 *
 * @param[in] job Whether to compute the eigenvectors. Case insensitive. Can be 'v' or 'n'.
 * @param[in] uplo Indicates how the data is stored. Case insensitive. Can be 'u' if the data is stored in the upper triangle, or 'l' if the
 * data is stored in the lower triangle.
 * @param[in] n The number of rows and columns in the matrix.
 * @param[inout] a The input matrix. On exit, it will be overwritten. If the eigenvectors are computed, then they will be stored in the
 * columns of this paramter.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] w The output for the eigenvalues.
 * @param[inout] work A work array. If @p lwork is -1, then the optimal work array size will be placed in the first element of the work
 * array.
 * @param[in] lwork The size of the work array. If it is -1, then a workspace query is performed and nothing else. The optimal workspace
 * size will then be placed in the first element of the work array.
 *
 * @return 0 on success. If positive, then the eigenvalue algorithm failed to converge. If negative, then one of the inputs had an invalid
 * value. The absolute value of the return indicates which parameter was invalid.
 *
 * @versionadded{2.0.0}
 */
auto cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork) -> int;
auto zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork) -> int;

/**
 * Computes the solution to system of linear equations @f$\mathbf{Ax} = \mathbf{B}@f$ for general
 * matrices.
 *
 * @param[in] n The number of rows and columns of the coefficient matrix.
 * @param[in] nrhs The number of columns in the right-hand side matrix.
 * @param[inout] a The coefficient matrix.
 * @param[in] lda The leading dimension of the coefficient matrix.
 * @param[out] ipiv A list for pivots used during LU decomposition.
 * @param[inout] b The right-hand side matrix.
 * @param[in] ldb The leading dimension of the right-hand side matrix.
 *
 * @return 0 on success. If positive, the coefficient matrix was singular. If negative, then there was an invalid parameter.
 * The absolute value indicates which parameter was invalid.
 * @versionadded{2.0.0}
 */
auto sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb, float *x, int ldx) -> int;
/// @copydoc sgesv
auto dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb, double *x, int ldx) -> int;
/// @copydoc sgesv
auto cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb, std::complex<float> *x, int ldx)
    -> int;
/// @copydoc sgesv
auto zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb, std::complex<double> *x, int ldx)
    -> int;

/**
 * Scales a vector by a scalar. Supports real scalars by real vectors, complex scalars by complex vectors,
 * and real scalars by complex vectors.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The scale factor.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The skip value for the vector.
 *
 * @versionadded{2.0.0}
 */
void sscal(int n, float alpha, float *vec, int inc);
/// @copydoc sscal
void dscal(int n, double alpha, double *vec, int inc);
/// @copydoc sscal
void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc);
/// @copydoc sscal
void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc);
/// @copydoc sscal
void csscal(int n, float alpha, std::complex<float> *vec, int inc);
/// @copydoc sscal
void zdscal(int n, double alpha, std::complex<double> *vec, int inc);

/**
 * Scales a vector by the reciprocal of a real value.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] alpha The denominator for the scale.
 * @param[inout] vec The vector to scale.
 * @param[in] inc The skip value for the vector.
 *
 * @versionadded{2.0.0}
 */
void srscl(int n, float alpha, float *vec, int inc);
void drscl(int n, double alpha, double *vec, int inc);
void csrscl(int n, float alpha, std::complex<float> *vec, int inc);
void zdrscl(int n, double alpha, std::complex<double> *vec, int inc);

/**
 * Compute the dot product between two vectors.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] x The left vector.
 * @param[in] incx The skip value for the left vector. If it is negative, the vector is traversed backwards.
 * @param[in] y The right vector.
 * @param[in] incy The skip value for the right vector. If it is negative, the vector is traversed backwards.
 *
 * @return The dot product between two vectors.
 *
 * @versionadded{2.0.0}
 */
auto sdot(int n, float const *x, int incx, float const *y, int incy) -> float;
/// @copydoc sdot
auto ddot(int n, double const *x, int incx, double const *y, int incy) -> double;
/**
 * Compute the unconjugated dot product between two vectors. The dot product is normally defined to be @f$ \sum_{i} x_i^*y_i@f$,
 * where the left vector is conjugated. However, this function performs @f$\sum_{i} x_iy_i@f$, akin to how it is done for
 * real vectors. For real vectors, the two definitions are the same. For complex vectors, the two definitions are different,
 * so keep this in mind when selecting between the two functions. This is the equivalent to BLAS's cdot<b>u</b>.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] x The left vector.
 * @param[in] incx The skip value for the left vector. If it is negative, the vector is traversed backwards.
 * @param[in] y The right vector.
 * @param[in] incy The skip value for the right vector. If it is negative, the vector is traversed backwards.
 *
 * @return The dot product between two vectors.
 *
 * @sa cdotc
 *
 * @versionadded{2.0.0}
 */
auto cdot(int n, std::complex<float> const *x, int incx, std::complex<float> const *y, int incy) -> std::complex<float>;
/// @copydoc cdot
auto zdot(int n, std::complex<double> const *x, int incx, std::complex<double> const *y, int incy) -> std::complex<double>;
/**
 * Compute the dot product between two vectors. The dot product is normally defined to be @f$ \sum_{i} x_i^*y_i@f$,
 * where the left vector is conjugated. This is how this function works. There is another way, where the first vector is not conjugated. For
 * that, use cdot. For real vectors, the two definitions are the same. For complex vectors, the two definitions are different, so keep this
 * in mind when selecting between the two functions.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] x The left vector.
 * @param[in] incx The skip value for the left vector. If it is negative, the vector is traversed backwards.
 * @param[in] y The right vector.
 * @param[in] incy The skip value for the right vector. If it is negative, the vector is traversed backwards.
 *
 * @return The dot product between two vectors.
 *
 * @versionadded{2.0.0}
 */
auto cdotc(int n, std::complex<float> const *x, int incx, std::complex<float> const *y, int incy) -> std::complex<float>;
/// @copydoc cdotc
auto zdotc(int n, std::complex<double> const *x, int incx, std::complex<double> const *y, int incy) -> std::complex<double>;

/**
 * Scale and add a vector to another. Performs @f$\mathbf{y} := \alpha\mathbf{x} + \mathbf{y}@f$.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha_x The scale factor for the input vector.
 * @param[in] x The input vector.
 * @param[in] inc_x The skip value for the input vector. If it is negative, the vector is traversed backwards.
 * @param[inout] y The output vector.
 * @param[in] inc_y The skip value for the output vector. If it is negative, the vector is traversed backwards.
 *
 * @versionadded{2.0.0}
 */
void saxpy(int n, float alpha_x, float const *x, int inc_x, float *y, int inc_y);
/// @copydoc saxpy
void daxpy(int n, double alpha_x, double const *x, int inc_x, double *y, int inc_y);
/// @copydoc saxpy
void caxpy(int n, std::complex<float> alpha_x, std::complex<float> const *x, int inc_x, std::complex<float> *y, int inc_y);
/// @copydoc saxpy
void zaxpy(int n, std::complex<double> alpha_x, std::complex<double> const *x, int inc_x, std::complex<double> *y, int inc_y);

/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{xy}^T + \mathbf{A}
 * @f]
 *
 * @param[in] m The number of elements in the left vector.
 * @param[in] n The number of elements in the right vector.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The left vector.
 * @param[in] inc_x The skip value for the left vector.
 * @param[in] y The right vector.
 * @param[in] inc_y The skip value for the right vector.
 * @param[out] a The matrix to update.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @throws std::domain_error If the dimensions are zero, or the leading dimension of the array is less than the number of columns.
 * @throws std::invalid_argument If either of the increment values is zero.
 *
 * @versionadded{2.0.0}
 */
void sger(int m, int n, float alpha, float const *x, int inc_x, float const *y, int inc_y, float *a, int lda);
/// @copydoc sger
void dger(int m, int n, double alpha, double const *x, int inc_x, double const *y, int inc_y, double *a, int lda);
/// @copydoc sger
void cger(int m, int n, std::complex<float> alpha, std::complex<float> const *x, int inc_x, std::complex<float> const *y, int inc_y,
          std::complex<float> *a, int lda);
/// @copydoc sger
void zger(int m, int n, std::complex<double> alpha, std::complex<double> const *x, int inc_x, std::complex<double> const *y, int inc_y,
          std::complex<double> *a, int lda);
/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 * @f[
 *    \mathbf{A} := \alpha\mathbf{xy}^H + \mathbf{A}
 * @f]
 *
 * @param[in] m The number of elements in the left vector.
 * @param[in] n The number of elements in the right vector.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The left vector.
 * @param[in] inc_x The skip value for the left vector.
 * @param[in] y The right vector.
 * @param[in] inc_y The skip value for the right vector.
 * @param[out] a The matrix to update.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @throws std::domain_error If the dimensions are zero, or the leading dimension of the array is less than the number of columns.
 * @throws std::invalid_argument If either of the increment values is zero.
 *
 * @versionadded{2.0.0}
 */
void cgerc(int m, int n, std::complex<float> alpha, std::complex<float> const *x, int inc_x, std::complex<float> const *y, int inc_y,
           std::complex<float> *a, int lda);
/// @copydoc cgerc
void zgerc(int m, int n, std::complex<double> alpha, std::complex<double> const *x, int inc_x, std::complex<double> const *y, int inc_y,
           std::complex<double> *a, int lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form
 * @f[
 *  \mathbf{A} = \mathbf{PLU}
 * @f]
 *
 * where @f$\mathbf{P}@f$ is a permutation matri, @f$\mathbf{L}@f$ is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and @f$\mathbf{U}@f$ is upper
 * triangular (upper trapezoidal if m < n).
 *
 * @param[in] m The number of rows of the matrix.
 * @param[in] n The number of columns of the matrix.
 * @param[inout] A The matrix to factor.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] ipiv The list of pivots used in the computation.
 *
 * @return 0 on success. If positive, the factorization completed successfully, but the matrix is singular so should not be used to solve a
 * system of equations. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{2.0.0}
 */
auto sgetrf(int m, int n, float *A, int lda, int *ipiv) -> int;
/// @copydoc sgetrf
auto dgetrf(int m, int n, double *A, int lda, int *ipiv) -> int;
/// @copydoc sgetrf
auto cgetrf(int m, int n, std::complex<float> *A, int lda, int *ipiv) -> int;
/// @copydoc sgetrf
auto zgetrf(int m, int n, std::complex<double> *A, int lda, int *ipiv) -> int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * @param[in] n The number of rows and columns of the matrix.
 * @param[in] A The matrix to invert after being processed by sgetrf.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] ipiv The pivots produced by sgetrf.
 * @param[out] C The inverse of the input matrix.
 * @param[in] ldc The leading dimension of the output matrix.
 *
 * @return 0 on success. If positive, then the matrix is singular and the inverse could not be computed.
 * If negative, then one of the parameters was invalid. The absolute value indicates which parameter was invalid.
 *
 * @versionadded{2.0.0}
 */
auto sgetri(int n, float const *A, int lda, int *ipiv, float *C, int ldc) -> int;
/// @copydoc sgetri
auto dgetri(int n, double const *A, int lda, int *ipiv, double *C, int ldc) -> int;
/// @copydoc sgetri
auto cgetri(int n, std::complex<float> const *A, int lda, int *ipiv, std::complex<float> *C, int ldc) -> int;
/// @copydoc sgetri
auto zgetri(int n, std::complex<double> const *A, int lda, int *ipiv, std::complex<double> *C, int ldc) -> int;

/**
 * Compute the sum of squares of a vector without overflow. The sum of squares will be <tt>scale * scale * sumsq</tt>.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] x The vector.
 * @param[in] incx The skip value for the vector.
 * @param[inout] scale The scale factor for the sum that avoids overflow/underflow. If it is non-zero, the @p sumsq will be considered as
 * part of the sum as well.
 * @param[inout] sumsq The scaled sum of squares. If it is non-zero, then it will be considered as part of the sum as well.
 *
 * @versionadded{2.0.0}
 */
void slassq(int n, float const *x, int incx, float *scale, float *sumsq);
/// @copydoc slassq
void dlassq(int n, double const *x, int incx, double *scale, double *sumsq);
/// @copydoc slassq
void classq(int n, std::complex<float> const *x, int incx, float *scale, float *sumsq);
/// @copydoc slassq
void zlassq(int n, std::complex<double> const *x, int incx, double *scale, double *sumsq);

/**
 * Compute the Euclidean norm of a vector.
 *
 * @param[in] n The number of elements in the vector.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The Euclidean norm of the vector.
 *
 * @versionadded{2.0.0}
 */
float snrm2(int n, float const *x, int incx);
/// @copydoc snrm2
double dnrm2(int n, double const *x, int incx);
/// @copydoc snrm2
float scnrm2(int n, std::complex<float> const *x, int incx);
/// @copydoc snrm2
double dznrm2(int n, std::complex<double> const *x, int incx);

/**
 * Compute the singular value decomposition using the QR algorithm.
 *
 * @param[in] jobu Whether to compute the U vectors. Case insensitive. Can be either 'a', 's', 'o', or 'n'.
 * @param[in] jobvt Whether to compute the V vectors. Case insensitive. Can be either 'a', 's', 'o', or 'n'.
 * @param[in] m The number of rows of the matrix.
 * @param[in] n The number of columns of the matrix.
 * @param[inout] a The matrix to decompose. Overwritten on exit.
 * @param[in] lda The leading dimension of the input matrix.
 * @param[out] s The singular values.
 * @param[out] u The U vectors.
 * @param[in] ldu The leading dimension of the U vectors. Must always be at least 1, even when not referenced.
 * @param[out] vt The transpose of the V vectors.
 * @param[in] ldvt The leading dimension of the transpose of the V vectors. Must always be at least 1, even when not referenced.
 * @param[out] superb Storage space for intermediates used in the calculation.
 *
 * @return 0 on success. If positive, then the algorithm did not converge. If negative, then one of the parameters had an invalid value.
 * The absolute value gives the parameter.
 *
 * @versionadded{2.0.0}
 */
auto sgesvd(char jobu, char jobvt, int m, int n, float *a, int lda, float *s, float *u, int ldu, float *vt, int ldvt, float *superb) -> int;
/// @copydoc sgesvd
auto dgesvd(char jobu, char jobvt, int m, int n, double *a, int lda, double *s, double *u, int ldu, double *vt, int ldvt, double *superb)
    -> int;
/// @copydoc sgesvd
auto cgesvd(char jobu, char jobvt, int m, int n, std::complex<float> *a, int lda, float *s, std::complex<float> *u, int ldu,
            std::complex<float> *vt, int ldvt, std::complex<float> *superb) -> int;
/// @copydoc sgesvd
auto zgesvd(char jobu, char jobvt, int m, int n, std::complex<double> *a, int lda, double *s, std::complex<double> *u, int ldu,
            std::complex<double> *vt, int ldvt, std::complex<double> *superb) -> int;

/**
 * Perform QR decomposition.
 *
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[inout] a The matrix to decompose. On exit, the R matrix is above the diagonal and the reflection vectors are below the diagonal.
 * @param[in] lda The leading dimension of the matrix.
 * @param[out] tau The scale factors for the Householder reflectors.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{2.0.0}
 */
auto sgeqrf(int m, int n, float *a, int lda, float *tau) -> int;
/// @copydoc sgeqrf
auto dgeqrf(int m, int n, double *a, int lda, double *tau) -> int;
/// @copydoc sgeqrf
auto cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int;
/// @copydoc sgeqrf
auto zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int;

/**
 * Extract the Q matrix from a QR decomposition.
 *
 * @param[in] m The number of rows in the input matrix.
 * @param[in] n The number of columns in the input matrix.
 * @param[in] k The number of elementary reflectors used to decompose the matrix.
 * @param[inout] a The input matrix after being processed by sgeqrf. On exit, it will contain the Q matrix.
 * @param[in] lda The leading dimension of the matrix.
 * @param[in] tau The scale factors for the elementary reflectors used in the decomposition.
 *
 * @return 0 on success. If negative, one of the parameters had an invalid value. The absolute value indicates which parameter was invalid.
 *
 * @versionadded{2.0.0}
 */
auto sorgqr(int m, int n, int k, float *a, int lda, float *tau) -> int;
/// @copydoc sorgqr
auto dorgqr(int m, int n, int k, double *a, int lda, double *tau) -> int;
/// @copydoc sorgqr
auto cungqr(int m, int n, int k, std::complex<float> *a, int lda, std::complex<float> *tau) -> int;
/// @copydoc sorgqr
auto zungqr(int m, int n, int k, std::complex<double> *a, int lda, std::complex<double> *tau) -> int;

/**
 * Copy one vector to another.
 *
 * @param[in] n The number of elements to copy.
 * @param[in] x The source vector.
 * @param[in] incx The skip value for the input. If zero, it will fill the output with the same value. If negative, then the input will be
 * traversed backwards.
 * @param[out] y The destination vector.
 * @param[in] incy The skip value for the destination vector. If it is negative, the destination will be traversed backwards.
 *
 * @versionadded{2.0.0}
 */
void scopy(int n, float const *x, int incx, float *y, int incy);
/// @copydoc scopy
void dcopy(int n, double const *x, int incx, double *y, int incy);
/// @copydoc scopy
void ccopy(int n, std::complex<float> const *x, int incx, std::complex<float> *y, int incy);
/// @copydoc scopy
void zcopy(int n, std::complex<double> const *x, int incx, std::complex<double> *y, int incy);

/**
 * Advanced matrix scaling operation that scales a matrix by <tt>cto/cfrom</tt> without overflow/underflow.
 *
 * @param[in] type The kind of matrix being processed. Case insensitive. Can be either 'g', 'l', 'u', 'h', 'b', 'q', or 'z'.
 * @param[in] kl The lower bandwidth of the matrix.
 * @param[in] ku The upper bandwidth of the matrix.
 * @param[in] cfrom The denominator for the scale.
 * @param[in] cto The numerator for the scale.
 * @param[in] m The number of rows in the matrix.
 * @param[in] n The number of columns in the matrix.
 * @param[out] vec The matrix to scale.
 * @param[in] lda The leading dimension of the matrix.
 *
 * @return 0 on success. If negative, then one of the parameters had an invalid value. The absolute value indicates which parameter was
 * invalid.
 *
 * @versionadded{2.0.0}
 */
int slascl(char type, int kl, int ku, float cfrom, float cto, int m, int n, float *vec, int lda);
/// @copydoc slascl
int dlascl(char type, int kl, int ku, double cfrom, double cto, int m, int n, double *vec, int lda);

/**
 * Perform the direct product between two vectors and add it to another.
 *
 * @param[in] n The number of elements in the vectors.
 * @param[in] alpha The scale factor for the product.
 * @param[in] x The first input vector.
 * @param[in] incx The skip value for the first vector. If negative, then it will be traversed backwards.
 * @param[in] y The second vector.
 * @param[in] incy The skip value for the second vector. If negative, then it will be traversed backwards.
 * @param[inout] z The vector to accumulate to.
 * @param[in] incz The skip value for the accumulation vector. If negative, then it will be traversed backwards.
 *
 * @versionadded{2.0.0}
 */
void sdirprod(int n, float alpha, float const *x, int incx, float const *y, int incy, float *z, int incz);
/// @copydoc sdirprod
void ddirprod(int n, double alpha, double const *x, int incx, double const *y, int incy, double *z, int incz);
/// @copydoc sdirprod
void cdirprod(int n, std::complex<float> alpha, std::complex<float> const *x, int incx, std::complex<float> const *y, int incy,
              std::complex<float> *z, int incz);
/// @copydoc sdirprod
void zdirprod(int n, std::complex<double> alpha, std::complex<double> const *x, int incx, std::complex<double> const *y, int incy,
              std::complex<double> *z, int incz);

/**
 * Computes the sum of the absolute values of the input vector. If the vector is complex,
 * then it is the sum of the absolute values of the components, not the magnitudes.
 *
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs as stated above.
 *
 * @versionadded{2.0.0}
 */
float sasum(int n, float const *x, int incx);
/// @copydoc sasum
double dasum(int n, double const *x, int incx);
/// @copydoc sasum
float scasum(int n, std::complex<float> const *x, int incx);
/// @copydoc sasum
double dzasum(int n, std::complex<double> const *x, int incx);

/**
 * Computes the sum of the absolute values of the input vector.
 *
 * @param[in] n The number of elements.
 * @param[in] x The vector to process.
 * @param[in] incx The skip value for the vector.
 *
 * @return The sum of the absolute values of the inputs.
 *
 * @versionadded{2.0.0}
 */
float scsum1(int n, std::complex<float> const *x, int incx);
/// @copydoc scsum1
double dzsum1(int n, std::complex<double> const *x, int incx);

/**
 * Conjugate a vector.
 *
 * @param[in] n The number of elements in the vector.
 * @param[inout] x The vector to conjugate.
 * @param[in] incx The skip value for the vector.
 */
void clacgv(int n, std::complex<float> *x, int incx);
/// @copydoc clacgv
void zlacgv(int n, std::complex<double> *x, int incx);
} // namespace einsums::blas::hip