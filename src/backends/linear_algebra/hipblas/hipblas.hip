#include "einsums/_GPUUtils.hpp"

#include <hip/driver_types.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/internal/hipsolver-types.h>
#include <memory>

#include "hipblas.hpp"

namespace einsums::backend::linear_algebra::hipblas {

using namespace einsums::gpu;

namespace detail {

hipblasHandle_t   blas_handle = nullptr;

__host__ __device__ hipblasOperation_t hipblas_char_to_op(char trans) {
    switch (trans) {
    case 't':
    case 'T':
        return HIPBLAS_OP_T;
    case 'c':
    case 'C':
        return HIPBLAS_OP_C;
    default:
        return HIPBLAS_OP_N;
    }
}

__host__ void hipblas_catch(hipblasStatus_t status, bool throw_success) {
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        if (throw_success) {
            throw detail::blasSuccess();
        } else {
            return;
        }
        break;
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        throw einsums::backend::linear_algebra::hipblas::detail::blasNotInitialized();
        break;
    case HIPBLAS_STATUS_ALLOC_FAILED:
        throw detail::blasAllocFailed();
        break;
    case HIPBLAS_STATUS_INVALID_VALUE:
        throw detail::blasInvalidValue();
        break;
    case HIPBLAS_STATUS_MAPPING_ERROR:
        throw detail::blasMappingError();
        break;
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        throw detail::blasExecutionFailed();
        break;
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        throw detail::blasInternalError();
        break;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        throw detail::blasNotSupported();
        break;
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        throw detail::blasArchMismatch();
        break;
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
        throw detail::blasHandleIsNullptr();
        break;
    case HIPBLAS_STATUS_INVALID_ENUM:
        throw detail::blasInvalidEnum();
        break;
    default:
        throw detail::blasUnknown();
    }
}

const char *hipsolverStatusToString(hipsolverStatus_t status) {
    switch (status) {
    case HIPSOLVER_STATUS_SUCCESS:
        return "HIPSOLVER_STATUS_SUCCESS: Success.";
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
        return "HIPSOLVER_STATUS_NOT_INITIALIZED: Handle has not been initialized. Make sure to call hipsolverCreate().";
    case HIPSOLVER_STATUS_ALLOC_FAILED:
        return "HIPSOLVER_STATUS_ALLOC_FAILED: Could not allocate resources.";
    case HIPSOLVER_STATUS_INVALID_VALUE:
        return "HIPSOLVER_STATUS_INVALID_VALUE: An unsupported numerical value was passed to a function.";
    case HIPSOLVER_STATUS_MAPPING_ERROR:
        return "HIPSOLVER_STATUS_MAPPING_ERROR: Could not access GPU memory space.";
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
        return "HIPSOLVER_STATUS_EXECUTION_FAILED: Failed to execute the program or function.";
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
        return "HIPSOLVER_STATUS_INTERNAL_ERROR: An unspecified internal error has occurred.";
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
        return "HIPSOLVER_STATUS_NOT_SUPPORTED: The function requested is not supported.";
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
        return "HIPSOLVER_STATUS_ARCH_MISMATCH: The code was compiled for a different device than it is being run on.";
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
        return "HIPSOLVER_STATUS_HANDLE_IS_NULLPTR: The handle that was passed to the function was the null pointer. Make sure it is "
               "initialized properly.";
    case HIPSOLVER_STATUS_INVALID_ENUM:
        return "HIPSOLVER_STATUS_INVALID_ENUM: An unsupported enum value was passed to the function.";
    case HIPSOLVER_STATUS_UNKNOWN:
        return "HIPSOLVER_STATUS_UNKNOWN: The backend returned an unsupported status code.";
    case HIPSOLVER_STATUS_ZERO_PIVOT:
        return "HIPSOLVER_STATUS_ZERO_PIVOT: A pivot of zero was chosen, leading to a zero-division.";
    default:
        return "Unrecognized status code from HIPSolver was passed to the stringifier. Assuming HIPSOLVER_STATUS_UNKNOWN.";
    }
}

hipblasHandle_t get_blas_handle() {
    return blas_handle;
}

hipblasHandle_t set_blas_handle(hipblasHandle_t value) {
    blas_handle = value;
    return blas_handle;
}

} // namespace detail

using namespace detail;

__host__ void initialize() {
    hip_catch(hipFree(nullptr));

    if (detail::blas_handle == nullptr) {
        hipblas_catch(hipblasCreate(&detail::blas_handle));
    }

    if (detail::get_solver_handle() == nullptr) {
        hipsolverHandle_t handle;
        hipsolver_catch(hipsolverCreate(&handle));
        detail::set_solver_handle(handle);
    }
}

__host__ void finalize() {
    if (detail::blas_handle != nullptr) {
        hipblas_catch(hipblasDestroy(detail::blas_handle));
        detail::blas_handle = nullptr;
    }

    if (detail::get_solver_handle() != nullptr) {
        hipsolver_catch(hipsolverDestroy(detail::get_solver_handle()));
        detail::set_solver_handle(nullptr);
    }
}

/*!
 * Performs matrix multiplication for general square matices of type double.
 */
__host__ void sgemm(char transa, char transb, int m, int n, int k, float alpha, const float *a, int lda, const float *b, int ldb,
                    float beta, float *c, int ldc) {

    // Alias to make copy-pasting easier.
    using math_type = float;

    // Create variables for the arrays.
    math_type *a_gpu, *b_gpu, *c_gpu;

    // Do this to save on a memory transfer.
    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {alpha, beta};

    // Calculate the sizes of the arrays.
    size_t a_size = transa ? k * lda : m * lda, b_size = transb ? ldb * n : ldb * k, c_size = m * ldc;

    // Allocate memory.
    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&c_gpu, c_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    // Copy data to the GPU.
    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    // If beta is zero, don't transfer C, since it is useless. Also, if ldc is not k, then there will be data overwritten. That is bad.
    if (beta != 0 || ldc != k) {
        hip_catch(hipMemcpy(c_gpu, c, c_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    // Compute.
    hipblas_catch(hipblasSgemm(detail::blas_handle, hipblas_char_to_op(transa), hipblas_char_to_op(transb), m, n, k, &(gpu_scalars->alpha),
                               a_gpu, lda, b_gpu, ldb, &(gpu_scalars->beta), c_gpu, ldc));

    // Wait for the device to finish.
    hip_catch(hipDeviceSynchronize());

    // Get the results back.
    hip_catch(hipMemcpy(c, c_gpu, c_size * sizeof(math_type), hipMemcpyDeviceToHost));

    // Free memory.
    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(c_gpu));
    hip_catch(hipFree(gpu_scalars));
}

__host__ void dgemm(char transa, char transb, int m, int n, int k, double alpha, const double *a, int lda, const double *b, int ldb,
                    double beta, double *c, int ldc) {
    // Alias to make copy-pasting easier.
    using math_type = double;

    // Create variables for the arrays.
    math_type *a_gpu, *b_gpu, *c_gpu;

    // Do this to save on a memory transfer.
    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {alpha, beta};

    // Calculate the sizes of the arrays.
    size_t a_size = transa ? k * lda : m * lda, b_size = transb ? ldb * n : ldb * k, c_size = m * ldc;

    // Allocate memory.
    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&c_gpu, c_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    // Copy data to the GPU.
    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    // If beta is zero, don't transfer C, since it is useless.
    if (beta != 0 || ldc != k) {
        hip_catch(hipMemcpy(c_gpu, c, c_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    // Compute.
    hipblas_catch(hipblasDgemm(detail::blas_handle, hipblas_char_to_op(transa), hipblas_char_to_op(transb), m, n, k, &(gpu_scalars->alpha),
                               a_gpu, lda, b_gpu, ldb, &(gpu_scalars->beta), c_gpu, ldc));

    // Wait for the device to finish.
    hip_catch(hipDeviceSynchronize());

    // Get the results back.
    hip_catch(hipMemcpy(c, c_gpu, c_size * sizeof(math_type), hipMemcpyDeviceToHost));

    // Free memory.
    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(c_gpu));
    hip_catch(hipFree(gpu_scalars));
}

__host__ void cgemm(char transa, char transb, int m, int n, int k, std::complex<float> alpha, const std::complex<float> *a, int lda,
                    const std::complex<float> *b, int ldb, std::complex<float> beta, std::complex<float> *c, int ldc) {
    // Alias to make copy-pasting easier.
    using math_type = hipblasComplex;

    // Create variables for the arrays.
    math_type *a_gpu, *b_gpu, *c_gpu;

    // Do this to save on a memory transfer.
    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {{alpha.real(), alpha.imag()}, {beta.real(), beta.imag()}};

    // Calculate the sizes of the arrays.
    size_t a_size = transa ? k * lda : m * lda, b_size = transb ? ldb * n : ldb * k, c_size = m * ldc;

    // Allocate memory.
    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&c_gpu, c_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    // Copy data to the GPU.
    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    // If beta is zero, don't transfer C, since it is useless.
    if (beta != 0.0f || ldc != k) {
        hip_catch(hipMemcpy(c_gpu, c, c_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    // Compute.
    hipblas_catch(hipblasCgemm(detail::blas_handle, hipblas_char_to_op(transa), hipblas_char_to_op(transb), m, n, k, &(gpu_scalars->alpha),
                               a_gpu, lda, b_gpu, ldb, &(gpu_scalars->beta), c_gpu, ldc));

    // Wait for the device to finish.
    hip_catch(hipDeviceSynchronize());

    // Get the results back.
    hip_catch(hipMemcpy(c, c_gpu, c_size * sizeof(math_type), hipMemcpyDeviceToHost));

    // Free memory.
    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(c_gpu));
    hip_catch(hipFree(gpu_scalars));
}

__host__ void zgemm(char transa, char transb, int m, int n, int k, std::complex<double> alpha, const std::complex<double> *a, int lda,
                    const std::complex<double> *b, int ldb, std::complex<double> beta, std::complex<double> *c, int ldc) {
    // Alias to make copy-pasting easier.
    using math_type = hipblasDoubleComplex;

    // Create variables for the arrays.
    math_type *a_gpu, *b_gpu, *c_gpu;

    // Do this to save on a memory transfer.
    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {{alpha.real(), alpha.imag()}, {beta.real(), beta.imag()}};

    // Calculate the sizes of the arrays.
    size_t a_size = transa ? k * lda : m * lda, b_size = transb ? ldb * n : ldb * k, c_size = m * ldc;

    // Allocate memory.
    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&c_gpu, c_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    // Copy data to the GPU.
    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    // If beta is zero, don't transfer C, since it is useless.
    if (beta != 0.0 || ldc != k) {
        hip_catch(hipMemcpy(c_gpu, c, c_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    // Compute.
    hipblas_catch(hipblasZgemm(detail::blas_handle, hipblas_char_to_op(transa), hipblas_char_to_op(transb), m, n, k, &(gpu_scalars->alpha),
                               a_gpu, lda, b_gpu, ldb, &(gpu_scalars->beta), c_gpu, ldc));

    // Wait for the device to finish.
    hip_catch(hipDeviceSynchronize());

    // Get the results back.
    hip_catch(hipMemcpy(c, c_gpu, c_size * sizeof(math_type), hipMemcpyDeviceToHost));

    // Free memory.
    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(c_gpu));
    hip_catch(hipFree(gpu_scalars));
}

/*!
 * Performs matrix vector multiplication.
 */
void sgemv(char transa, int m, int n, float alpha, const float *a, int lda, const float *x, int incx, float beta, float *y, int incy) {
    using math_type = float;

    math_type *a_gpu, *x_gpu, *y_gpu;

    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {alpha, beta};

    size_t a_size = transa ? n * lda : m * lda, x_size = transa ? m * incx : n * incx, y_size = transa ? n * incy : m * incy;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    if (beta != 0 || incy != 1) {
        hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    hipblas_catch(hipblasSgemv(blas_handle, hipblas_char_to_op(transa), m, n, &(gpu_scalars->alpha), a_gpu, lda, x_gpu, incx,
                               &(gpu_scalars->beta), y_gpu, incy));

    hip_catch(hipMemcpy(y, y_gpu, y_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));
    hip_catch(hipFree(gpu_scalars));
}

void dgemv(char transa, int m, int n, double alpha, const double *a, int lda, const double *x, int incx, double beta, double *y, int incy) {
    using math_type = double;

    math_type *a_gpu, *x_gpu, *y_gpu;

    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {alpha, beta};

    size_t a_size = transa ? n * lda : m * lda, x_size = transa ? m * incx : n * incx, y_size = transa ? n * incy : m * incy;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    if (beta != 0 || incy != 1) {
        hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    hipblas_catch(hipblasDgemv(blas_handle, hipblas_char_to_op(transa), m, n, &(gpu_scalars->alpha), a_gpu, lda, x_gpu, incx,
                               &(gpu_scalars->beta), y_gpu, incy));

    hip_catch(hipMemcpy(y, y_gpu, y_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));
    hip_catch(hipFree(gpu_scalars));
}

void cgemv(char transa, int m, int n, std::complex<float> alpha, const std::complex<float> *a, int lda, const std::complex<float> *x,
           int incx, std::complex<float> beta, std::complex<float> *y, int incy) {
    using math_type = hipblasComplex;

    math_type *a_gpu, *x_gpu, *y_gpu;

    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {{alpha.real(), alpha.imag()}, {beta.real(), beta.imag()}};

    size_t a_size = transa ? n * lda : m * lda, x_size = transa ? m * incx : n * incx, y_size = transa ? n * incy : m * incy;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    if (beta != 0.0f || incy != 1) {
        hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    hipblas_catch(hipblasCgemv(blas_handle, hipblas_char_to_op(transa), m, n, &(gpu_scalars->alpha), a_gpu, lda, x_gpu, incx,
                               &(gpu_scalars->beta), y_gpu, incy));

    hip_catch(hipMemcpy(y, y_gpu, y_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));
    hip_catch(hipFree(gpu_scalars));
}
void zgemv(char transa, int m, int n, std::complex<double> alpha, const std::complex<double> *a, int lda, const std::complex<double> *x,
           int incx, std::complex<double> beta, std::complex<double> *y, int incy) {
    using math_type = hipblasDoubleComplex;

    math_type *a_gpu, *x_gpu, *y_gpu;

    struct scalars_t {
        math_type alpha, beta;
    } *gpu_scalars, cpu_scalars = {{alpha.real(), alpha.imag()}, {beta.real(), beta.imag()}};

    size_t a_size = transa ? n * lda : m * lda, x_size = transa ? m * incx : n * incx, y_size = transa ? n * incy : m * incy;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));
    hip_catch(hipMalloc(&gpu_scalars, sizeof(scalars_t)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(gpu_scalars, &cpu_scalars, sizeof(scalars_t), hipMemcpyHostToDevice));

    if (beta != 0.0 || incy != 1) {
        hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));
    }

    hipblas_catch(hipblasZgemv(blas_handle, hipblas_char_to_op(transa), m, n, &(gpu_scalars->alpha), a_gpu, lda, x_gpu, incx,
                               &(gpu_scalars->beta), y_gpu, incy));

    hip_catch(hipMemcpy(y, y_gpu, y_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));
    hip_catch(hipFree(gpu_scalars));
}

void sscal(int n, float alpha, float *vec, int inc) {
    using math_type = float;

    math_type *vec_gpu, *alpha_gpu;

    size_t vec_size = n * inc;

    hip_catch(hipMalloc(&vec_gpu, vec_size * sizeof(math_type)));
    hip_catch(hipMalloc(&alpha_gpu, sizeof(math_type)));

    hip_catch(hipMemcpy(vec_gpu, vec, vec_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(math_type), hipMemcpyHostToDevice));

    hipblas_catch(hipblasSscal(blas_handle, n, alpha_gpu, vec_gpu, inc));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(vec, vec_gpu, vec_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(vec_gpu));
    hip_catch(hipFree(alpha_gpu));
}

void dscal(int n, double alpha, double *vec, int inc) {
    using math_type = double;

    math_type *vec_gpu, *alpha_gpu;

    size_t vec_size = n * inc;

    hip_catch(hipMalloc(&vec_gpu, vec_size * sizeof(math_type)));
    hip_catch(hipMalloc(&alpha_gpu, sizeof(math_type)));

    hip_catch(hipMemcpy(vec_gpu, vec, vec_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(math_type), hipMemcpyHostToDevice));

    hipblas_catch(hipblasDscal(blas_handle, n, alpha_gpu, vec_gpu, inc));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(vec, vec_gpu, vec_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(vec_gpu));
    hip_catch(hipFree(alpha_gpu));
}

void cscal(int n, std::complex<float> alpha, std::complex<float> *vec, int inc) {
    using math_type = hipblasComplex;

    math_type *vec_gpu, *alpha_gpu;

    size_t vec_size = n * inc;

    hip_catch(hipMalloc(&vec_gpu, vec_size * sizeof(math_type)));
    hip_catch(hipMalloc(&alpha_gpu, sizeof(math_type)));

    hip_catch(hipMemcpy(vec_gpu, vec, vec_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(math_type), hipMemcpyHostToDevice));

    hipblas_catch(hipblasCscal(blas_handle, n, alpha_gpu, vec_gpu, inc));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(vec, vec_gpu, vec_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(vec_gpu));
    hip_catch(hipFree(alpha_gpu));
}

void zscal(int n, std::complex<double> alpha, std::complex<double> *vec, int inc) {
    using math_type = hipblasDoubleComplex;

    math_type *vec_gpu, *alpha_gpu;

    size_t vec_size = n * inc;

    hip_catch(hipMalloc(&vec_gpu, vec_size * sizeof(math_type)));
    hip_catch(hipMalloc(&alpha_gpu, sizeof(math_type)));

    hip_catch(hipMemcpy(vec_gpu, vec, vec_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(alpha_gpu, &alpha, sizeof(math_type), hipMemcpyHostToDevice));

    hipblas_catch(hipblasZscal(blas_handle, n, alpha_gpu, vec_gpu, inc));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(vec, vec_gpu, vec_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(vec_gpu));
    hip_catch(hipFree(alpha_gpu));
}

float sdot(int n, const float *x, int incx, const float *y, int incy) {
    using math_type = float;

    math_type result, *x_gpu, *y_gpu;

    size_t x_size = n * incx, y_size = n * incy;

    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));

    // result can be device or host, according to the docs.
    hipblas_catch(hipblasSdot(blas_handle, n, x_gpu, incx, y_gpu, incy, &result));

    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));

    return result;
}

double ddot(int n, const double *x, int incx, const double *y, int incy) {
    using math_type = double;

    math_type result, *x_gpu, *y_gpu;

    size_t x_size = n * incx, y_size = n * incy;

    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));

    // result can be device or host, according to the docs.
    hipblas_catch(hipblasDdot(blas_handle, n, x_gpu, incx, y_gpu, incy, &result));

    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));

    return result;
}

std::complex<float> cdot(int n, const std::complex<float> *x, int incx, const std::complex<float> *y, int incy) {
    using math_type = hipblasComplex;

    math_type result, *x_gpu, *y_gpu;

    size_t x_size = n * incx, y_size = n * incy;

    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));

    // result can be device or host, according to the docs.
    hipblas_catch(hipblasCdotc(blas_handle, n, x_gpu, incx, y_gpu, incy, &result));

    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));

    return std::complex<float>(result.real(), result.imag());
}

std::complex<double> zdot(int n, const std::complex<double> *x, int incx, const std::complex<double> *y, int incy) {
    using math_type = hipblasDoubleComplex;

    math_type result, *x_gpu, *y_gpu;

    size_t x_size = n * incx, y_size = n * incy;

    hip_catch(hipMalloc(&x_gpu, x_size * sizeof(math_type)));
    hip_catch(hipMalloc(&y_gpu, y_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, x, x_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(y_gpu, y, y_size * sizeof(math_type), hipMemcpyHostToDevice));

    // result can be device or host, according to the docs.
    hipblas_catch(hipblasZdotc(blas_handle, n, x_gpu, incx, y_gpu, incy, &result));

    hip_catch(hipFree(x_gpu));
    hip_catch(hipFree(y_gpu));

    return std::complex<double>(result.real(), result.imag());
}

void saxpy(int n, float alpha_x, const float *x, int inc_x, float *y, int inc_y);
void daxpy(int n, double alpha_x, const double *x, int inc_x, double *y, int inc_y);
void caxpy(int n, std::complex<float> alpha_x, const std::complex<float> *x, int inc_x, std::complex<float> *y, int inc_y);
void zaxpy(int n, std::complex<double> alpha_x, const std::complex<double> *x, int inc_x, std::complex<double> *y, int inc_y);

/*!
 * Performs a rank-1 update of a general matrix.
 *
 * The ?ger routines perform a matrix-vector operator defined as
 *    A := alpha*x*y' + A,
 * where:
 *   alpha is a scalar
 *   x is an m-element vector,
 *   y is an n-element vector,
 *   A is an m-by-n general matrix
 */
void dger(int m, int n, double alpha, const double *x, int inc_x, const double *y, int inc_y, double *a, int lda);

/*!
 * Computes the LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form
 *   A = P * L * U
 * where P is a permutation matri, L is lower triangular with
 * unit diagonal elements (lower trapezoidal if m > n) and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 */
auto dgetrf(int, int, double *, int, int *) -> int;

/*!
 * Computes the inverse of a matrix using the LU factorization computed
 * by getrf
 *
 * Returns INFO
 *   0 if successful
 *  <0 the (-INFO)-th argument has an illegal value
 *  >0 U(INFO, INFO) is exactly zero; the matrix is singular
 */
auto dgetri(int, double *, int, const int *, double *, int) -> int;

auto dlange(char norm_type, int m, int n, const double *A, int lda, double *work) -> double;

auto sgesvd(char, char, int, int, float *, int, float *, float *, int, float *, int, float *) -> int;
auto dgesvd(char, char, int, int, double *, int, double *, double *, int, double *, int, double *) -> int;

auto sgesdd(char, int, int, float *, int, float *, float *, int, float *, int) -> int;
auto dgesdd(char, int, int, double *, int, double *, double *, int, double *, int) -> int;
auto cgesdd(char, int, int, std::complex<float> *, int, float *, std::complex<float> *, int, std::complex<float> *, int) -> int;
auto zgesdd(char, int, int, std::complex<double> *, int, double *, std::complex<double> *, int, std::complex<double> *, int) -> int;

auto sgees(char jobvs, int n, float *a, int lda, int *sdim, float *wr, float *wi, float *vs, int ldvs) -> int;
auto dgees(char jobvs, int n, double *a, int lda, int *sdim, double *wr, double *wi, double *vs, int ldvs) -> int;

auto strsyl(char trana, char tranb, int isgn, int m, int n, const float *a, int lda, const float *b, int ldb, float *c, int ldc,
            float *scale) -> int;
auto dtrsyl(char trana, char tranb, int isgn, int m, int n, const double *a, int lda, const double *b, int ldb, double *c, int ldc,
            double *scale) -> int;
auto ctrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<float> *a, int lda, const std::complex<float> *b, int ldb,
            std::complex<float> *c, int ldc, float *scale) -> int;
auto ztrsyl(char trana, char tranb, int isgn, int m, int n, const std::complex<double> *a, int lda, const std::complex<double> *b, int ldb,
            std::complex<double> *c, int ldc, double *scale) -> int;

auto sgeqrf(int m, int n, float *a, int lda, float *tau) -> int;
auto dgeqrf(int m, int n, double *a, int lda, double *tau) -> int;
auto cgeqrf(int m, int n, std::complex<float> *a, int lda, std::complex<float> *tau) -> int;
auto zgeqrf(int m, int n, std::complex<double> *a, int lda, std::complex<double> *tau) -> int;

auto sorgqr(int m, int n, int k, float *a, int lda, const float *tau) -> int;
auto dorgqr(int m, int n, int k, double *a, int lda, const double *tau) -> int;
auto cungqr(int m, int n, int k, std::complex<float> *a, int lda, const std::complex<float> *tau) -> int;
auto zungqr(int m, int n, int k, std::complex<double> *a, int lda, const std::complex<double> *tau) -> int;

} // namespace einsums::backend::linear_algebra::hipblas