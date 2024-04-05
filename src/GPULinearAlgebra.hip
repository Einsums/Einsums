#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include "einsums/GPULinearAlgebra.hpp"

#include <hip/driver_types.h>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

using namespace einsums::gpu;
using namespace einsums::gpu::detail;

namespace einsums {
namespace linear_algebra {


namespace detail {

    namespace gpu {

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float *alpha, const float *a, int lda,
          const float *b, int ldb, const float *beta, float *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double *alpha, const double *a, int lda,
          const double *b, int ldb, const double *beta, double *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipComplex *alpha, const hipComplex *a, int lda,
          const hipComplex *b, int ldb, const hipComplex *beta, hipComplex *c, int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgemm(handle, transa, transb, m, n, k, (const hipblasComplex *)alpha, (const hipblasComplex *)a, lda,
                               (const hipblasComplex *)b, ldb, (const hipblasComplex *)beta, (hipblasComplex *)c, ldc));
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const hipDoubleComplex *alpha,
          const hipDoubleComplex *a, int lda, const hipDoubleComplex *b, int ldb, const hipDoubleComplex *beta, hipDoubleComplex *c,
          int ldc) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgemm(handle, transa, transb, m, n, k, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)a, lda,
                               (const hipblasDoubleComplex *)b, ldb, (const hipblasDoubleComplex *)beta, (hipblasDoubleComplex *)c, ldc));
}

void ger(int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

void ger(int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

void ger(int m, int n, const hipComplex *alpha, const hipComplex *x, int incx, const hipComplex *y, int incy, hipComplex *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgerc(handle, m, n, (const hipblasComplex *)alpha, (const hipblasComplex *)x, incx, (const hipblasComplex *)y,
                               incy, (hipblasComplex *)A, lda));
}

void ger(int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *x, int incx, const hipDoubleComplex *y, int incy,
         hipDoubleComplex *A, int lda) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgerc(handle, m, n, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)x, incx,
                               (const hipblasDoubleComplex *)y, incy, (hipblasDoubleComplex *)A, lda));
}

void gemv(hipblasOperation_t transa, int m, int n, const float *alpha, const float *a, int lda, const float *x, int incx, const float *beta,
          float *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(hipblasOperation_t transa, int m, int n, const double *alpha, const double *a, int lda, const double *x, int incx,
          const double *beta, double *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(hipblasOperation_t transa, int m, int n, const hipComplex *alpha, const hipComplex *a, int lda, const hipComplex *x, int incx,
          const hipComplex *beta, hipComplex *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCgemv(handle, transa, m, n, (const hipblasComplex *)alpha, (const hipblasComplex *)a, lda,
                               (const hipblasComplex *)x, incx, (const hipblasComplex *)beta, (hipblasComplex *)y, incy));
}

void gemv(hipblasOperation_t transa, int m, int n, const hipDoubleComplex *alpha, const hipDoubleComplex *a, int lda,
          const hipDoubleComplex *x, int incx, const hipDoubleComplex *beta, hipDoubleComplex *y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZgemv(handle, transa, m, n, (const hipblasDoubleComplex *)alpha, (const hipblasDoubleComplex *)a, lda,
                               (const hipblasDoubleComplex *)x, incx, (const hipblasDoubleComplex *)beta, (hipblasDoubleComplex *)y, incy));
}

void scal(int size, const float *alpha, float *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSscal(handle, size, alpha, x, incx));
}

void scal(int size, const double *alpha, double *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDscal(handle, size, alpha, x, incx));
}

void scal(int size, const hipComplex *alpha, hipComplex *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCscal(handle, size, (const hipblasComplex *)alpha, (hipblasComplex *)x, incx));
}

void scal(int size, const hipDoubleComplex *alpha, hipDoubleComplex *x, int incx) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZscal(handle, size, (const hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)x, incx));
}

int gesv(int n, int nrhs, float *A, int lda, int *ipiv, float *B, int ldb, float *X, int ldx) {

    hipsolverHandle_t handle = get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, get_stream()));

    int niters;

    hipsolver_catch(hipsolverSSgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, get_stream()));

    stream_wait();

    hip_catch(hipFreeAsync(out_gpu, get_stream()));
    hip_catch(hipFreeAsync(work, get_stream()));


    return out;
}
int gesv(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, double *X, int ldx) {
    hipsolverHandle_t handle = get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, get_stream()));

    int niters;

    hipsolver_catch(hipsolverDDgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, get_stream()));

    stream_wait();

    hip_catch(hipFreeAsync(out_gpu, get_stream()));
    hip_catch(hipFreeAsync(work, get_stream()));

    return out;
}

int gesv(int n, int nrhs, hipComplex *A, int lda, int *ipiv, hipComplex *B, int ldb, hipComplex *X, int ldx) {
    hipsolverHandle_t handle = get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, get_stream()));

    int niters;

    hipsolver_catch(hipsolverCCgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, get_stream()));

    stream_wait();

    hip_catch(hipFreeAsync(out_gpu, get_stream()));
    hip_catch(hipFreeAsync(work, get_stream()));

    return out;
}
int gesv(int n, int nrhs, hipDoubleComplex *A, int lda, int *ipiv, hipDoubleComplex *B, int ldb, hipDoubleComplex *X, int ldx) {
    hipsolverHandle_t handle = get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, get_stream()));

    int niters;

    hipsolver_catch(hipsolverZZgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, get_stream()));

    stream_wait();

    hip_catch(hipFreeAsync(out_gpu, get_stream()));
    hip_catch(hipFreeAsync(work, get_stream()));

    return out;
}

EINSUMS_EXPORT void axpy(int n, const float *alpha, const float *X, int incx, float *Y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasSaxpy(handle, n, alpha, X, incx, Y, incy));
}

EINSUMS_EXPORT void axpy(int n, const double *alpha, const double *X, int incx, double *Y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasDaxpy(handle, n, alpha, X, incx, Y, incy));
}
EINSUMS_EXPORT void axpy(int n, const hipComplex *alpha, const hipComplex *X, int incx, hipComplex *Y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasCaxpy(handle, n, (hipblasComplex *)alpha, (hipblasComplex *)X, incx, (hipblasComplex *)Y, incy));
}
EINSUMS_EXPORT void axpy(int n, const hipDoubleComplex *alpha, const hipDoubleComplex *X, int incx, hipDoubleComplex *Y, int incy) {
    hipblasHandle_t handle = get_blas_handle();

    hipblas_catch(hipblasZaxpy(handle, n, (hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)X, incx, (hipblasDoubleComplex *)Y, incy));
}

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const float *A, int lda, const float *B, int ldb, float *C,
                                         int ldc) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread, max_size = m * m * n * n;

    while (curr_index < max_size) {
        size_t a, b, i, j;

        size_t quotient = curr_index;

        a = quotient % n;
        quotient /= n;
        b = quotient % n;
        quotient /= n;
        i = quotient % m;
        quotient /= m;
        j = quotient;

        curr_index += kernel_size;
        float term;

        if (!TransA) {
            term = A[i * lda + j];
        } else {
            term = A[j * lda + i];
        }

        if (!TransB) {
            term *= B[i * ldb + a] * B[j * ldb + b];
        } else {
            term *= B[a * ldb + i] * B[b * ldb + j];
        }

        atomicAdd(C + (a * ldc + b), term);
    }
}

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const double *A, int lda, const double *B, int ldb,
                                         double *C, int ldc) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread, max_size = m * m * n * n;

    while (curr_index < max_size) {
        size_t a, b, i, j;

        size_t quotient = curr_index;

        a = quotient % n;
        quotient /= n;
        b = quotient % n;
        quotient /= n;
        i = quotient % m;
        quotient /= m;
        j = quotient;

        curr_index += kernel_size;
        double term;

        if (!TransA) {
            term = A[i * lda + j];
        } else {
            term = A[j * lda + i];
        }

        if (!TransB) {
            term *= B[i * ldb + a] * B[j * ldb + b];
        } else {
            term *= B[a * ldb + i] * B[b * ldb + j];
        }

        atomicAdd(C + (a * ldc + b), term);
    }
}

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const hipComplex *A, int lda, const hipComplex *B, int ldb,
                                         hipComplex *C, int ldc) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread, max_size = m * m * n * n;

    while (curr_index < max_size) {
        size_t a, b, i, j;

        size_t quotient = curr_index;

        a = quotient % n;
        quotient /= n;
        b = quotient % n;
        quotient /= n;
        i = quotient % m;
        quotient /= m;
        j = quotient;

        curr_index += kernel_size;
        hipComplex term;

        if (!TransA) {
            term = A[i * lda + j];
        } else {
            term = A[j * lda + i];
        }

        if (!TransB) {
            term = term * B[i * ldb + a] * B[j * ldb + b];
        } else {
            term = term * B[a * ldb + i] * B[b * ldb + j];
        }

        atomicAdd(&(C[a * ldc + b].x), term.x);
        atomicAdd(&(C[a * ldc + b].y), term.y);
    }
}

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, const hipDoubleComplex *A, int lda,
                                         const hipDoubleComplex *B, int ldb, hipDoubleComplex *C, int ldc) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread, max_size = m * m * n * n;

    while (curr_index < max_size) {
        size_t a, b, i, j;

        size_t quotient = curr_index;

        a = quotient % n;
        quotient /= n;
        b = quotient % n;
        quotient /= n;
        i = quotient % m;
        quotient /= m;
        j = quotient;

        curr_index += kernel_size;
        hipDoubleComplex term;

        if (!TransA) {
            term = A[i * lda + j];
        } else {
            term = A[j * lda + i];
        }

        if (!TransB) {
            term = term * B[i * ldb + a] * B[j * ldb + b];
        } else {
            term = term * B[a * ldb + i] * B[b * ldb + j];
        }

        atomicAdd(&(C[a * ldc + b].x), term.x);
        atomicAdd(&(C[a * ldc + b].y), term.y);
    }
}

int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, float *A, int lda, float *D) {
    hipsolverHandle_t handle = get_solver_handle();

    int *dev_info;

    int out;

    int    lwork;
    float *work;

    hipsolver_catch(hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork));

    hip_catch(hipMallocAsync((void **)&work, lwork * sizeof(float), get_stream()));

    hip_catch(hipMallocAsync((void **)&dev_info, sizeof(int), get_stream()));

    hipsolver_catch(hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, dev_info));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)dev_info, sizeof(int), hipMemcpyDeviceToHost, get_stream()));

    stream_wait();

    hip_catch(hipFreeAsync(dev_info, get_stream()));

    return out;
}

int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double *A, int lda, double *D) {
    hipsolverHandle_t handle = get_solver_handle();

    int *dev_info;

    int out;

    int     lwork;
    double *work;

    hipsolver_catch(hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork));

    hip_catch(hipMallocAsync((void **)&work, lwork * sizeof(double), get_stream()));

    hip_catch(hipMallocAsync((void **)&dev_info, sizeof(int), get_stream()));

    hipsolver_catch(hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, dev_info));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)dev_info, sizeof(int), hipMemcpyDeviceToHost, get_stream()));
    
    stream_wait();

    hip_catch(hipFreeAsync(dev_info, get_stream()));

    return out;
}

    }

} // namespace detail

}
}
