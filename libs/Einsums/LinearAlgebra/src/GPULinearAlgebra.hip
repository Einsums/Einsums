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

__global__ EINSUMS_EXPORT void symm_gemm(bool TransA, bool TransB, int m, int n, hipFloatComplex const *A, int lda,
                                         hipFloatComplex const *B, int ldb, hipFloatComplex *C, int ldc) {
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
        hipFloatComplex term;

        if (!TransA) {
            term = A[i * lda + j];
        } else {
            term = A[j * lda + i];
        }

        if (!TransB) {
            term = gpu_ops::mult(gpu_ops::mult(term, B[i * ldb + a]), B[j * ldb + b]);
        } else {
            term = gpu_ops::mult(gpu_ops::mult(term, B[a * ldb + i]), B[b * ldb + j]);
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
            term = gpu_ops::mult(gpu_ops::mult(term, B[i * ldb + a]), B[j * ldb + b]);
        } else {
            term = gpu_ops::mult(gpu_ops::mult(term, B[a * ldb + i]), B[b * ldb + j]);
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