#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUStreams/GPUStreams.hpp>
#include <Einsums/LinearAlgebra/GPULinearAlgebra.hpp>

#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <omp.h>
#include <vector>

namespace einsums::linear_algebra {
namespace detail {

namespace gpu {

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, float const *alpha, float const *a, int lda,
          float const *b, int ldb, float const *beta, float *c, int ldc) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasSgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, double const *alpha, double const *a, int lda,
          double const *b, int ldb, double const *beta, double *c, int ldc) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasDgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc));
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, hipComplex const *alpha, hipComplex const *a, int lda,
          hipComplex const *b, int ldb, hipComplex const *beta, hipComplex *c, int ldc) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasCgemm(handle, transa, transb, m, n, k, (hipblasComplex const *)alpha, (hipblasComplex const *)a, lda,
                               (hipblasComplex const *)b, ldb, (hipblasComplex const *)beta, (hipblasComplex *)c, ldc));
}

void gemm(hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, hipDoubleComplex const *alpha,
          hipDoubleComplex const *a, int lda, hipDoubleComplex const *b, int ldb, hipDoubleComplex const *beta, hipDoubleComplex *c,
          int ldc) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasZgemm(handle, transa, transb, m, n, k, (hipblasDoubleComplex const *)alpha, (hipblasDoubleComplex const *)a, lda,
                               (hipblasDoubleComplex const *)b, ldb, (hipblasDoubleComplex const *)beta, (hipblasDoubleComplex *)c, ldc));
}

void ger(int m, int n, float const *alpha, float const *x, int incx, float const *y, int incy, float *A, int lda) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

void ger(int m, int n, double const *alpha, double const *x, int incx, double const *y, int incy, double *A, int lda) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

void ger(int m, int n, hipComplex const *alpha, hipComplex const *x, int incx, hipComplex const *y, int incy, hipComplex *A, int lda) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasCgerc(handle, m, n, (hipblasComplex const *)alpha, (hipblasComplex const *)x, incx, (hipblasComplex const *)y,
                               incy, (hipblasComplex *)A, lda));
}

void ger(int m, int n, hipDoubleComplex const *alpha, hipDoubleComplex const *x, int incx, hipDoubleComplex const *y, int incy,
         hipDoubleComplex *A, int lda) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasZgerc(handle, m, n, (hipblasDoubleComplex const *)alpha, (hipblasDoubleComplex const *)x, incx,
                               (hipblasDoubleComplex const *)y, incy, (hipblasDoubleComplex *)A, lda));
}

void gemv(hipblasOperation_t transa, int m, int n, float const *alpha, float const *a, int lda, float const *x, int incx, float const *beta,
          float *y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasSgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(hipblasOperation_t transa, int m, int n, double const *alpha, double const *a, int lda, double const *x, int incx,
          double const *beta, double *y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasDgemv(handle, transa, m, n, alpha, a, lda, x, incx, beta, y, incy));
}

void gemv(hipblasOperation_t transa, int m, int n, hipComplex const *alpha, hipComplex const *a, int lda, hipComplex const *x, int incx,
          hipComplex const *beta, hipComplex *y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasCgemv(handle, transa, m, n, (hipblasComplex const *)alpha, (hipblasComplex const *)a, lda,
                               (hipblasComplex const *)x, incx, (hipblasComplex const *)beta, (hipblasComplex *)y, incy));
}

void gemv(hipblasOperation_t transa, int m, int n, hipDoubleComplex const *alpha, hipDoubleComplex const *a, int lda,
          hipDoubleComplex const *x, int incx, hipDoubleComplex const *beta, hipDoubleComplex *y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasZgemv(handle, transa, m, n, (hipblasDoubleComplex const *)alpha, (hipblasDoubleComplex const *)a, lda,
                               (hipblasDoubleComplex const *)x, incx, (hipblasDoubleComplex const *)beta, (hipblasDoubleComplex *)y, incy));
}

void scal(int size, float const *alpha, float *x, int incx) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasSscal(handle, size, alpha, x, incx));
}

void scal(int size, double const *alpha, double *x, int incx) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasDscal(handle, size, alpha, x, incx));
}

void scal(int size, hipComplex const *alpha, hipComplex *x, int incx) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasCscal(handle, size, (hipblasComplex const *)alpha, (hipblasComplex *)x, incx));
}

void scal(int size, hipDoubleComplex const *alpha, hipDoubleComplex *x, int incx) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasZscal(handle, size, (hipblasDoubleComplex const *)alpha, (hipblasDoubleComplex *)x, incx));
}

int gesv(int n, int nrhs, float *A, int lda, int *ipiv, float *B, int ldb, float *X, int ldx) {

    hipsolverHandle_t handle = einsums::gpu::get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), einsums::gpu::get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, einsums::gpu::get_stream()));

    int niters;

    hipsolver_catch(hipsolverSSgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, einsums::gpu::get_stream()));

    einsums::gpu::stream_wait();

    hip_catch(hipFreeAsync(out_gpu, einsums::gpu::get_stream()));
    hip_catch(hipFreeAsync(work, einsums::gpu::get_stream()));

    return out;
}
int gesv(int n, int nrhs, double *A, int lda, int *ipiv, double *B, int ldb, double *X, int ldx) {
    hipsolverHandle_t handle = einsums::gpu::get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), einsums::gpu::get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, einsums::gpu::get_stream()));

    int niters;

    hipsolver_catch(hipsolverDDgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, einsums::gpu::get_stream()));

    einsums::gpu::stream_wait();

    hip_catch(hipFreeAsync(out_gpu, einsums::gpu::get_stream()));
    hip_catch(hipFreeAsync(work, einsums::gpu::get_stream()));

    return out;
}

int gesv(int n, int nrhs, hipComplex *A, int lda, int *ipiv, hipComplex *B, int ldb, hipComplex *X, int ldx) {
    hipsolverHandle_t handle = einsums::gpu::get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), einsums::gpu::get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, einsums::gpu::get_stream()));

    int niters;

    hipsolver_catch(hipsolverCCgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, einsums::gpu::get_stream()));

    einsums::gpu::stream_wait();

    hip_catch(hipFreeAsync(out_gpu, einsums::gpu::get_stream()));
    hip_catch(hipFreeAsync(work, einsums::gpu::get_stream()));

    return out;
}
int gesv(int n, int nrhs, hipDoubleComplex *A, int lda, int *ipiv, hipDoubleComplex *B, int ldb, hipDoubleComplex *X, int ldx) {
    hipsolverHandle_t handle = einsums::gpu::get_solver_handle();

    int  out;
    int *out_gpu;

    hip_catch(hipMallocAsync((void **)&out_gpu, sizeof(int), einsums::gpu::get_stream()));

    size_t work_size;

    hipsolver_catch(hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, &work_size));

    void *work;

    hip_catch(hipMallocAsync(&work, work_size, einsums::gpu::get_stream()));

    int niters;

    hipsolver_catch(hipsolverZZgesv(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, work_size, &niters, out_gpu));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)out_gpu, sizeof(int), hipMemcpyDeviceToHost, einsums::gpu::get_stream()));

    einsums::gpu::stream_wait();

    hip_catch(hipFreeAsync(out_gpu, einsums::gpu::get_stream()));
    hip_catch(hipFreeAsync(work, einsums::gpu::get_stream()));

    return out;
}

EINSUMS_EXPORT void axpy(int n, float const *alpha, float const *X, int incx, float *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasSaxpy(handle, n, alpha, X, incx, Y, incy));
}

EINSUMS_EXPORT void axpy(int n, double const *alpha, double const *X, int incx, double *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasDaxpy(handle, n, alpha, X, incx, Y, incy));
}
EINSUMS_EXPORT void axpy(int n, hipComplex const *alpha, hipComplex const *X, int incx, hipComplex *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasCaxpy(handle, n, (hipblasComplex *)alpha, (hipblasComplex *)X, incx, (hipblasComplex *)Y, incy));
}
EINSUMS_EXPORT void axpy(int n, hipDoubleComplex const *alpha, hipDoubleComplex const *X, int incx, hipDoubleComplex *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    hipblas_catch(hipblasZaxpy(handle, n, (hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)X, incx, (hipblasDoubleComplex *)Y, incy));
}

EINSUMS_EXPORT void axpby(int n, float const *alpha, float const *X, int incx, float const *beta, float *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    // hipblas_catch(hipblasSaxpby(handle, n, alpha, X, incx, beta, Y, incy));
    hipStream_t stream;
    hipblas_catch(hipblasGetStream(handle, &stream));
    saxpby_kernel<<<einsums::gpu::block_size(n), einsums::gpu::blocks(n), 0, stream>>>(n, alpha, X, incx, beta, Y, incy);
}

EINSUMS_EXPORT void axpby(int n, double const *alpha, double const *X, int incx, double const *beta, double *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    // hipblas_catch(hipblasDaxpby(handle, n, alpha, X, incx, beta, Y, incy));

    hipStream_t stream;
    hipblas_catch(hipblasGetStream(handle, &stream));
    daxpby_kernel<<<einsums::gpu::block_size(n), einsums::gpu::blocks(n), 0, stream>>>(n, alpha, X, incx, beta, Y, incy);
}
EINSUMS_EXPORT void axpby(int n, hipComplex const *alpha, hipComplex const *X, int incx, hipComplex const *beta, hipComplex *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    // hipblas_catch(hipblasCaxpby(handle, n, (hipblasComplex *)alpha, (hipblasComplex *)X, incx, beta, (hipblasComplex *)Y, incy));

    hipStream_t stream;
    hipblas_catch(hipblasGetStream(handle, &stream));
    caxpby_kernel<<<einsums::gpu::block_size(n), einsums::gpu::blocks(n), 0, stream>>>(n, alpha, X, incx, beta, Y, incy);
}
EINSUMS_EXPORT void axpby(int n, hipDoubleComplex const *alpha, hipDoubleComplex const *X, int incx, hipDoubleComplex const *beta,
                          hipDoubleComplex *Y, int incy) {
    hipblasHandle_t handle = einsums::gpu::get_blas_handle();

    // hipblas_catch(
    //     hipblasZaxpby(handle, n, (hipblasDoubleComplex *)alpha, (hipblasDoubleComplex *)X, incx, beta, (hipblasDoubleComplex *)Y, incy));

    hipStream_t stream;
    hipblas_catch(hipblasGetStream(handle, &stream));
    zaxpby_kernel<<<einsums::gpu::block_size(n), einsums::gpu::blocks(n), 0, stream>>>(n, alpha, X, incx, beta, Y, incy);
}

__global__ EINSUMS_EXPORT void saxpby_kernel(int n, float const *alpha, float const *X, int incx, float const *beta, float *Y, int incy) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread;

    for (size_t curr_index = thread; curr_index < n; curr_index += kernel_size) {
        size_t x_index = incx * curr_index;
        size_t y_index = incy * curr_index;

        Y[y_index] = *alpha * X[x_index] + *beta * Y[y_index];
    }
}
__global__ EINSUMS_EXPORT void daxpby_kernel(int n, double const *alpha, double const *X, int incx, double const *beta, double *Y,
                                             int incy) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread;

    for (size_t curr_index = thread; curr_index < n; curr_index += kernel_size) {
        size_t x_index = incx * curr_index;
        size_t y_index = incy * curr_index;

        Y[y_index] = *alpha * X[x_index] + *beta * Y[y_index];
    }
}
__global__ EINSUMS_EXPORT void caxpby_kernel(int n, hipComplex const *alpha, hipComplex const *X, int incx, hipComplex const *beta,
                                             hipComplex *Y, int incy) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread;

    for (size_t curr_index = thread; curr_index < n; curr_index += kernel_size) {
        size_t x_index = incx * curr_index;
        size_t y_index = incy * curr_index;

        Y[y_index] = *alpha * X[x_index] + *beta * Y[y_index];
    }
}
__global__ EINSUMS_EXPORT void zaxpby_kernel(int n, hipDoubleComplex const *alpha, hipDoubleComplex const *X, int incx,
                                             hipDoubleComplex const *beta, hipDoubleComplex *Y, int incy) {
    int thread, kernel_size;

    get_worker_info(thread, kernel_size);

    size_t curr_index = thread;

    for (size_t curr_index = thread; curr_index < n; curr_index += kernel_size) {
        size_t x_index = incx * curr_index;
        size_t y_index = incy * curr_index;

        Y[y_index] = *alpha * X[x_index] + *beta * Y[y_index];
    }
}

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, float const *A, int lda, float const *B, int ldb, float *C,
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

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, double const *A, int lda, double const *B, int ldb,
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

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipComplex const *A, int lda, hipComplex const *B, int ldb,
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

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipDoubleComplex const *A, int lda,
                                         hipDoubleComplex const *B, int ldb, hipDoubleComplex *C, int ldc) {
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
    hipsolverHandle_t handle = einsums::gpu::get_solver_handle();

    int *dev_info;

    int out;

    int    lwork;
    float *work;

    hipsolver_catch(hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork));

    hip_catch(hipMallocAsync((void **)&work, lwork * sizeof(float), einsums::gpu::get_stream()));

    hip_catch(hipMallocAsync((void **)&dev_info, sizeof(int), einsums::gpu::get_stream()));

    hipsolver_catch(hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, dev_info));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)dev_info, sizeof(int), hipMemcpyDeviceToHost, einsums::gpu::get_stream()));

    einsums::gpu::stream_wait();

    hip_catch(hipFreeAsync(dev_info, einsums::gpu::get_stream()));

    return out;
}

int syev(hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, double *A, int lda, double *D) {
    hipsolverHandle_t handle = einsums::gpu::get_solver_handle();

    int *dev_info;

    int out;

    int     lwork;
    double *work;

    hipsolver_catch(hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, &lwork));

    hip_catch(hipMallocAsync((void **)&work, lwork * sizeof(double), einsums::gpu::get_stream()));

    hip_catch(hipMallocAsync((void **)&dev_info, sizeof(int), einsums::gpu::get_stream()));

    hipsolver_catch(hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, dev_info));

    hip_catch(hipMemcpyAsync((void *)&out, (void *)dev_info, sizeof(int), hipMemcpyDeviceToHost, einsums::gpu::get_stream()));

    einsums::gpu::stream_wait();

    hip_catch(hipFreeAsync(dev_info, einsums::gpu::get_stream()));

    return out;
}

} // namespace gpu

} // namespace detail

} // namespace einsums::linear_algebra