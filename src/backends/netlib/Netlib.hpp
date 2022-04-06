#pragma once

namespace einsums::backend::netlib {

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb, double beta,
           double *c, int ldc);

} // namespace einsums::backend::netlib