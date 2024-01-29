#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/../../backends/linear_algebra/hipblas/hipblas.hpp"
#include "einsums/GPULinearAlgebra.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

using namespace einsums::gpu;
using namespace einsums::gpu::detail;
using namespace einsums::backend::linear_algebra::hipblas;
using namespace einsums::backend::linear_algebra::hipblas::detail;

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::gpu::linear_algebra)

namespace detail {

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *a, int lda,
                 const float *b, int ldb, const float *beta, float *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));

    hip_catch(hipDeviceSynchronize());
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *a, int lda,
                  const double *b, int ldb, const double *beta, double *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));

    hip_catch(hipDeviceSynchronize());
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipComplex *alpha,
                      const hipComplex *a, int lda, const hipComplex *b, int ldb, const hipComplex *beta, hipComplex *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgemm(handle, transa, transb, m, n, k, (const hipblasComplex *)alpha, (const hipblasComplex *)a, lda,
                               (const hipblasComplex *)b, ldb, (const hipblasComplex *)beta, (hipblasComplex *)c, ldc));

    hip_catch(hipDeviceSynchronize());
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipDoubleComplex *alpha,
                            const hipDoubleComplex *a, int lda, const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta,
                            hipDoubleComplex *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgemm(handle, transa, transb, m, n, k, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)a, lda,
                               (const hipblasDoubleComplex *)b, ldb, (const hipblasDoubleComplex *)beta, (hipblasDoubleComplex *)c, ldc));

    hip_catch(hipDeviceSynchronize());
}

void ger(int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));

    hip_catch(hipDeviceSynchronize());
}

void ger(int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda));

    hip_catch(hipDeviceSynchronize());
}

void ger(int m, int n, const hipComplex *alpha, const hipComplex *x, int incx, const hipComplex *y, int incy, hipComplex *A,
                     int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgerc(handle, m, n, (const hipblasComplex *)alpha, (const hipblasComplex *)x, incx, (const hipblasComplex *)y,
                               incy, (hipblasComplex *)A, lda));

    hip_catch(hipDeviceSynchronize());
}

void ger(int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *x, int incx, const hipDoubleComplex *y,
                           int incy, hipDoubleComplex *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgerc(handle, m, n, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)x, incx,
                               (const hipblasDoubleComplex *)y, incy, (hipblasDoubleComplex *)A, lda));

    hip_catch(hipDeviceSynchronize());
}

void gemv(hipblasOperation_t transa, int m, int n, const float *alpha, const float *a, int lda, const float *x, int incx,
                 const float *beta, float *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));

    hip_catch(hipDeviceSynchronize());
}

void gemv(hipblasOperation_t transa, int m, int n, const double *alpha, const double *a, int lda, const double *x, int incx,
                  const double *beta, double *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));

    hip_catch(hipDeviceSynchronize());
}

void gemv(hipblasOperation_t transa, int m, int n, const hipComplex *alpha, const hipComplex *a, int lda, const hipComplex *x,
                      int incx, const hipComplex *beta, hipComplex *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgemv(handle, transa, m, n, (const hipblasComplex *)alpha, (const hipblasComplex *)a, lda,
                               (const hipblasComplex *)x, incx, (const hipblasComplex *)beta, (hipblasComplex *)y, incy));

    hip_catch(hipDeviceSynchronize());
}

void gemv(hipblasOperation_t transa, int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
                            const hipDoubleComplex *x, int incx, const hipDoubleComplex *beta, hipDoubleComplex *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgemv(handle, transa, m, n, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)a, lda,
                               (const hipblasDoubleComplex *)x, incx, (const hipblasDoubleComplex *)beta, (hipblasDoubleComplex *)y, incy));

    hip_catch(hipDeviceSynchronize());
}

void scal(int size, const float *alpha, float *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSscal(handle, size, alpha, x, incx));

    hip_catch(hipDeviceSynchronize());
}

void scal(int size, const double *alpha, double *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDscal(handle, size, alpha, x, incx));

    hip_catch(hipDeviceSynchronize());
}

void scal(int size, const hipComplex *alpha, hipComplex *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCscal(handle, size, (const hipblasComplex *)alpha, (hipblasComplex *)x, incx));

    hip_catch(hipDeviceSynchronize());
}

void scal(int size, const hipDoubleComplex *alpha, hipDoubleComplex *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZscal(handle, size, (const hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)x, incx));

    hip_catch(hipDeviceSynchronize());
}

}

END_EINSUMS_NAMESPACE_CPP(einsums::gpu::linear_algebra)
