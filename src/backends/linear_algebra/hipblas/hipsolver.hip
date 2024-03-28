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

hipsolverHandle_t solver_handle = nullptr;

__host__ __device__ hipsolverOperation_t hipsolver_char_to_op(char trans) {
    switch (trans) {
    case 't':
    case 'T':
        return HIPSOLVER_OP_T;
    case 'c':
    case 'C':
        return HIPSOLVER_OP_C;
    default:
        return HIPSOLVER_OP_N;
    }
}

__host__ __device__ hipsolverEigMode_t hipsolver_job(char job) {
    switch (job) {
    case 'n':
    case 'N':
        return HIPSOLVER_EIG_MODE_NOVECTOR;
    case 'v':
    case 'V':
        return HIPSOLVER_EIG_MODE_VECTOR;
    default:
        return HIPSOLVER_EIG_MODE_NOVECTOR;
    }
}

__host__ __device__ hipsolverFillMode_t hipsolver_fill(char fill) {
    switch (fill) {
    case 'u':
    case 'U':
        return HIPSOLVER_FILL_MODE_UPPER;
    case 'l':
    case 'L':
        return HIPSOLVER_FILL_MODE_LOWER;
    default:
        return HIPSOLVER_FILL_MODE_UPPER;
    }
}

__host__ void __hipsolver_catch__(hipsolverStatus_t condition, const char *diagnostic, bool throw_success) {
    switch (condition) {
    case HIPSOLVER_STATUS_SUCCESS:
        if (throw_success) {
            throw detail::solverSuccess(diagnostic);
        } else {
            return;
        }
        break;
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
        throw detail::solverNotInitialized(diagnostic);
        break;
    case HIPSOLVER_STATUS_ALLOC_FAILED:
        throw detail::solverAllocFailed(diagnostic);
        break;
    case HIPSOLVER_STATUS_INVALID_VALUE:
        throw detail::solverInvalidValue(diagnostic);
        break;
    case HIPSOLVER_STATUS_MAPPING_ERROR:
        throw detail::solverMappingError(diagnostic);
        break;
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
        throw detail::solverExecutionFailed(diagnostic);
        break;
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
        throw detail::solverInternalError(diagnostic);
        break;
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
        throw detail::solverFuncNotSupported(diagnostic);
        break;
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
        throw detail::solverArchMismatch(diagnostic);
        break;
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
        throw detail::solverHandleIsNullptr(diagnostic);
        break;
    case HIPSOLVER_STATUS_INVALID_ENUM:
        throw detail::solverInvalidEnum(diagnostic);
        break;
    case HIPSOLVER_STATUS_UNKNOWN:
        throw detail::solverUnknown(diagnostic);
        break;
    // case HIPSOLVER_STATUS_ZERO_PIVOT:
    //     throw detail::solverZeroPivot();
    //     break;
    default:
        throw detail::solverUnknown(diagnostic);
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
    // case HIPSOLVER_STATUS_ZERO_PIVOT:
    //     return "HIPSOLVER_STATUS_ZERO_PIVOT: A pivot of zero was chosen, leading to a zero-division.";
    default:
        return "Unrecognized status code from HIPSolver was passed to the stringifier. Assuming HIPSOLVER_STATUS_UNKNOWN.";
    }
}

hipsolverHandle_t get_solver_handle() {
    return solver_handle;
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value) {
    solver_handle = value;
    return solver_handle;
}

} // namespace detail

using namespace detail;

/*!
 * Performs symmetric matrix diagonalization.
 */
int ssyev(char job, char uplo, int n, float *a, int lda, float *w, float *work, int lwork) {

    using math_type = float;

    int *info = new int();

    math_type *a_gpu, *w_gpu;

    size_t a_size = n * lda, w_size = n;

    bool work_registered = false;

    // Check for work properties.
    if (lwork != 0 && work != nullptr) {
        hipPointerAttribute_t attribs;
        hipError_t            status = hipPointerGetAttributes(&attribs, work);

        if (status == hipErrorInvalidValue) {
            hip_catch(hipHostRegister(work, lwork * sizeof(math_type), hipHostRegisterDefault));
            work_registered = true;
        } else if (status != hipSuccess) {
            hip_catch(status);
        }
    }

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&w_gpu, w_size * sizeof(math_type)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    if (lwork == 0 || work == nullptr) {
        hipsolver_catch(hipsolverSsyevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n, a_gpu, lda,
                                        w_gpu, nullptr, 0, info));
    } else {
        hipsolver_catch(hipsolverSsyevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n, a_gpu, lda,
                                        w_gpu, work, lwork, info));
    }

    hip_catch(hipDeviceSynchronize());

    int out = *info;

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(w, w_gpu, w_size * sizeof(math_type), hipMemcpyDeviceToHost));

    if (work_registered) {
        hip_catch(hipHostUnregister(work));
    }

    hip_catch(hipHostUnregister(info));
    delete info;

    return out;
}

int dsyev(char job, char uplo, int n, double *a, int lda, double *w, double *work, int lwork) {

    using math_type = double;

    int *info = new int();

    math_type *a_gpu, *w_gpu;

    size_t a_size = n * lda, w_size = n;

    bool work_registered = false;
    // Check for work properties.
    if (lwork != 0 && work != nullptr) {
        hipPointerAttribute_t attribs;
        hipError_t            status = hipPointerGetAttributes(&attribs, work);

        if (status == hipErrorInvalidValue) {
            hip_catch(hipHostRegister(work, lwork * sizeof(math_type), hipHostRegisterDefault));
            work_registered = true;
        } else if (status != hipSuccess) {
            hip_catch(status);
        }
    }

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&w_gpu, w_size * sizeof(math_type)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    if (lwork == 0 || work == nullptr) {
        hipsolver_catch(hipsolverDsyevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n, a_gpu, lda,
                                        w_gpu, nullptr, 0, info));
    } else {
        hipsolver_catch(hipsolverDsyevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n, a_gpu, lda,
                                        w_gpu, work, lwork, info));
    }

    hip_catch(hipDeviceSynchronize());

    int out = *info;

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(w, w_gpu, w_size * sizeof(math_type), hipMemcpyDeviceToHost));

    if (work_registered) {
        hip_catch(hipHostUnregister(work));
    }

    hip_catch(hipHostUnregister(info));
    delete info;

    return out;
}

/*!
 * Computes all eigenvalues and, optionally, eigenvectors of a Hermitian matrix.
 */
int cheev(char job, char uplo, int n, std::complex<float> *a, int lda, float *w, std::complex<float> *work, int lwork, float *) {
    using math_type = hipFloatComplex;
    using scal_type = float;

    int *info = new int();

    math_type *a_gpu;
    scal_type *w_gpu;

    size_t a_size = n * lda, w_size = n;

    bool work_registered = false;

    // Check for work properties.
    if (lwork != 0 && work != nullptr) {
        hipPointerAttribute_t attribs;
        hipError_t            status = hipPointerGetAttributes(&attribs, work);

        if (status == hipErrorInvalidValue) {
            hip_catch(hipHostRegister(work, lwork * sizeof(math_type), hipHostRegisterDefault));
            work_registered = true;
        } else if (status != hipSuccess) {
            hip_catch(status);
        }
    }

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&w_gpu, w_size * sizeof(scal_type)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    if (lwork == 0 || work == nullptr) {
        hipsolver_catch(hipsolverCheevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n,
                                        (math_type *)a_gpu, lda, w_gpu, nullptr, 0, info));
    } else {
        hipsolver_catch(hipsolverCheevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n,
                                        (math_type *)a_gpu, lda, w_gpu, (math_type *)work, lwork, info));
    }

    hip_catch(hipDeviceSynchronize());

    int out = *info;

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(w, w_gpu, w_size * sizeof(scal_type), hipMemcpyDeviceToHost));

    if (work_registered) {
        hip_catch(hipHostUnregister(work));
    }

    hip_catch(hipHostUnregister(info));
    delete info;

    return out;
}

int zheev(char job, char uplo, int n, std::complex<double> *a, int lda, double *w, std::complex<double> *work, int lwork, double *) {
    using math_type = hipDoubleComplex;
    using scal_type = double;

    int *info = new int();

    math_type *a_gpu;
    scal_type *w_gpu;

    size_t a_size = n * lda, w_size = n;

    bool work_registered = false;

    // Check for work properties.
    if (lwork != 0 && work != nullptr) {
        hipPointerAttribute_t attribs;
        hipError_t            status = hipPointerGetAttributes(&attribs, work);

        if (status == hipErrorInvalidValue) {
            hip_catch(hipHostRegister(work, lwork * sizeof(math_type), hipHostRegisterDefault));
            work_registered = true;
        } else if (status != hipSuccess) {
            hip_catch(status);
        }
    }

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&w_gpu, w_size * sizeof(scal_type)));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    if (lwork == 0 || work == nullptr) {
        hipsolver_catch(hipsolverZheevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n,
                                        (math_type *)a_gpu, lda, w_gpu, nullptr, 0, info));
    } else {
        hipsolver_catch(hipsolverZheevd(detail::solver_handle, detail::hipsolver_job(job), detail::hipsolver_fill(uplo), n,
                                        (math_type *)a_gpu, lda, w_gpu, (math_type *)work, lwork, info));
    }

    hip_catch(hipDeviceSynchronize());

    int out = *info;

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(w, w_gpu, w_size * sizeof(scal_type), hipMemcpyDeviceToHost));

    if (work_registered) {
        hip_catch(hipHostUnregister(work));
    }

    hip_catch(hipHostUnregister(info));
    delete info;

    return out;
}

/*!
 * Computes the solution to system of linear equations A * x = B for general
 * matrices.
 */
int sgesv(int n, int nrhs, float *a, int lda, int *ipiv, float *b, int ldb) {
    using math_type = float;

    int *info = new int(), *ipiv_gpu;

    math_type *a_gpu, *b_gpu, *x_gpu;

    size_t a_size = n * lda, b_size = n * ldb;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Here there is a bit of divergence. Rocm can do in-place gesv like is defined by the prototype.
     * Cuda, however, does not support this. To avoid extra copies, split the paths.
     * HIP says that for Rocm, if B and X are the same pointer, the result will be as normal, simply
     * using the in-place version. This should take advantage of this fact.
     */
#ifdef __HIP_PLATFORM_AMD__
    // niters is nullptr so that if it does become used, it will break and we will get bug reports.
    hipsolver_catch(hipsolverSSgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, b_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, b_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hip_catch(hipMalloc(&x_gpu, b_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    hipsolver_catch(hipsolverSSgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, x_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, x_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(x_gpu));
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));

    int out = *info;

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(ipiv_gpu));

    hip_catch(hipHostUnregister(info));

    delete info;

    return out;
}

int dgesv(int n, int nrhs, double *a, int lda, int *ipiv, double *b, int ldb) {
    using math_type = double;

    int *info = new int(), *ipiv_gpu;

    math_type *a_gpu, *b_gpu, *x_gpu;

    size_t a_size = n * lda, b_size = n * ldb;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Here there is a bit of divergence. Rocm can do in-place gesv like is defined by the prototype.
     * Cuda, however, does not support this. To avoid extra copies, split the paths.
     * HIP says that for Rocm, if B and X are the same pointer, the result will be as normal, simply
     * using the in-place version. This should take advantage of this fact.
     */
#ifdef __HIP_PLATFORM_AMD__
    // niters is nullptr so that if it does become used, it will break and we will get bug reports.
    hipsolver_catch(hipsolverDDgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, b_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, b_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hip_catch(hipMalloc(&x_gpu, b_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    hipsolver_catch(hipsolverDDgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, x_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, x_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(x_gpu));
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));

    int out = *info;

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(ipiv_gpu));

    hip_catch(hipHostUnregister(info));

    delete info;

    return out;
}

int cgesv(int n, int nrhs, std::complex<float> *a, int lda, int *ipiv, std::complex<float> *b, int ldb) {
    using math_type = hipFloatComplex;

    int *info = new int(), *ipiv_gpu;

    math_type *a_gpu, *b_gpu, *x_gpu;

    size_t a_size = n * lda, b_size = n * ldb;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Here there is a bit of divergence. Rocm can do in-place gesv like is defined by the prototype.
     * Cuda, however, does not support this. To avoid extra copies, split the paths.
     * HIP says that for Rocm, if B and X are the same pointer, the result will be as normal, simply
     * using the in-place version. This should take advantage of this fact.
     */
#ifdef __HIP_PLATFORM_AMD__
    // niters is nullptr so that if it does become used, it will break and we will get bug reports.
    hipsolver_catch(hipsolverCCgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, b_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, b_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hip_catch(hipMalloc(&x_gpu, b_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    hipsolver_catch(hipsolverCCgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, x_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, x_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(x_gpu));
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));

    int out = *info;

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(ipiv_gpu));

    hip_catch(hipHostUnregister(info));

    delete info;

    return out;
}

int zgesv(int n, int nrhs, std::complex<double> *a, int lda, int *ipiv, std::complex<double> *b, int ldb) {
    using math_type = hipDoubleComplex;

    int *info = new int(), *ipiv_gpu;

    math_type *a_gpu, *b_gpu, *x_gpu;

    size_t a_size = n * lda, b_size = n * ldb;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));
    hip_catch(hipMalloc(&b_gpu, b_size * sizeof(math_type)));
    hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    hip_catch(hipHostRegister(info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));
    hip_catch(hipMemcpy(b_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Here there is a bit of divergence. Rocm can do in-place gesv like is defined by the prototype.
     * Cuda, however, does not support this. To avoid extra copies, split the paths.
     * HIP says that for Rocm, if B and X are the same pointer, the result will be as normal, simply
     * using the in-place version. This should take advantage of this fact.
     */
#ifdef __HIP_PLATFORM_AMD__
    // niters is nullptr so that if it does become used, it will break and we will get bug reports.
    hipsolver_catch(hipsolverZZgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, b_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, b_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hip_catch(hipMalloc(&x_gpu, b_size * sizeof(math_type)));

    hip_catch(hipMemcpy(x_gpu, b, b_size * sizeof(math_type), hipMemcpyHostToDevice));

    hipsolver_catch(hipsolverZZgesv(solver_handle, n, nrhs, a_gpu, lda, ipiv_gpu, b_gpu, ldb, x_gpu, ldb, nullptr, 0, nullptr, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipMemcpy(b, x_gpu, b_size * sizeof(math_type), hipMemcpyDeviceToHost));

    hip_catch(hipFree(x_gpu));
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));

    int out = *info;

    hip_catch(hipFree(a_gpu));
    hip_catch(hipFree(b_gpu));
    hip_catch(hipFree(ipiv_gpu));

    hip_catch(hipHostUnregister(info));

    delete info;

    return out;
}

int sgetrf(int m, int n, float *a, int lda, int *ipiv) {
    using math_type = float;

    hipsolverHandle_t handle = get_solver_handle();

    math_type *a_gpu, *work;
    int *ipiv_gpu;

    int lwork, *info = new int{0};

    size_t a_size = m * lda;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));

    if (ipiv != nullptr) {
        hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    } else {
        ipiv_gpu = nullptr;
    }

    hip_catch(hipHostRegister(&info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Cuda does not require a work array, and the argument is ignored. AMD requires a work array.
     */
#ifdef __HIP_PLATFORM_AMD__
    hipsolver_catch(hipsolverSgetrf_bufferSize(handle, m, n, a_gpu, lda, &lwork));

    hip_catch(hipMalloc(&work, lwork * sizeof(math_type)));

    hipsolver_catch(hipsolverSgetrf(handle, m, n, a_gpu, lda, work, lwork, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipFree(work));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hipsolver_catch(hipsolverSgetrf(handle, m, n, a_gpu, lda, nullptr, 0, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipFree(a_gpu));

    if (ipiv != nullptr) {
        hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));
        hip_catch(hipFree(ipiv_gpu));
    }

    hip_catch(hipHostUnregister(info));

    int out = *info;

    delete info;

    return out;
}

int dgetrf(int m, int n, double *a, int lda, int *ipiv) {
    using math_type = double;

    hipsolverHandle_t handle = get_solver_handle();

    math_type *a_gpu, *work;

    int *ipiv_gpu;

    int lwork, *info = new int{0};

    size_t a_size = m * lda;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));

    if (ipiv != nullptr) {
        hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    } else {
        ipiv_gpu = nullptr;
    }

    hip_catch(hipHostRegister(&info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Cuda does not require a work array, and the argument is ignored. AMD requires a work array.
     */
#ifdef __HIP_PLATFORM_AMD__
    hipsolver_catch(hipsolverDgetrf_bufferSize(handle, m, n, a_gpu, lda, &lwork));

    hip_catch(hipMalloc(&work, lwork * sizeof(math_type)));

    hipsolver_catch(hipsolverDgetrf(handle, m, n, a_gpu, lda, work, lwork, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipFree(work));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hipsolver_catch(hipsolverDgetrf(handle, m, n, a_gpu, lda, nullptr, 0, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipFree(a_gpu));

    if (ipiv != nullptr) {
        hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));
        hip_catch(hipFree(ipiv_gpu));
    }

    hip_catch(hipHostUnregister(info));

    int out = *info;

    delete info;

    return out;
}

int cgetrf(int m, int n, std::complex<float> *a, int lda, int *ipiv) {
    using math_type = hipblasComplex;

    hipsolverHandle_t handle = get_solver_handle();

    math_type *a_gpu, *work;

    int *ipiv_gpu;

    int lwork, *info = new int{0};

    size_t a_size = m * lda;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));

    if (ipiv != nullptr) {
        hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    } else {
        ipiv_gpu = nullptr;
    }

    hip_catch(hipHostRegister(&info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Cuda does not require a work array, and the argument is ignored. AMD requires a work array.
     */
#ifdef __HIP_PLATFORM_AMD__
    hipsolver_catch(hipsolverCgetrf_bufferSize(handle, m, n, (hipComplex *) a_gpu, lda, &lwork));

    hip_catch(hipMalloc(&work, lwork * sizeof(math_type)));

    hipsolver_catch(hipsolverCgetrf(handle, m, n, (hipComplex *) a_gpu, lda, (hipComplex *) work, lwork, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipFree(work));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hipsolver_catch(hipsolverCgetrf(handle, m, n, (hipComplex *) a_gpu, lda, nullptr, 0, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipFree(a_gpu));

    if (ipiv != nullptr) {
        hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));
        hip_catch(hipFree(ipiv_gpu));
    }

    hip_catch(hipHostUnregister(info));

    int out = *info;

    delete info;

    return out;
}

int zgetrf(int m, int n, std::complex<double> *a, int lda, int *ipiv) {
    using math_type = hipblasDoubleComplex;

    hipsolverHandle_t handle = get_solver_handle();

    math_type *a_gpu, *work;

    int *ipiv_gpu;

    int lwork, *info = new int{0};

    size_t a_size = m * lda;

    hip_catch(hipMalloc(&a_gpu, a_size * sizeof(math_type)));

    if (ipiv != nullptr) {
        hip_catch(hipMalloc(&ipiv_gpu, n * sizeof(int)));
    } else {
        ipiv_gpu = nullptr;
    }

    hip_catch(hipHostRegister(&info, sizeof(int), hipHostRegisterDefault));

    hip_catch(hipMemcpy(a_gpu, a, a_size * sizeof(math_type), hipMemcpyHostToDevice));

    /*
     * Cuda does not require a work array, and the argument is ignored. AMD requires a work array.
     */
#ifdef __HIP_PLATFORM_AMD__
    hipsolver_catch(hipsolverZgetrf_bufferSize(handle, m, n, (hipDoubleComplex *) a_gpu, lda, &lwork));

    hip_catch(hipMalloc(&work, lwork * sizeof(math_type)));

    hipsolver_catch(hipsolverZgetrf(handle, m, n, (hipDoubleComplex *) a_gpu, lda, (hipDoubleComplex *) work, lwork, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());

    hip_catch(hipFree(work));
#elif defined(__HIP_PLATFORM_NVIDIA__)
    hipsolver_catch(hipsolverZgetrf(handle, m, n, a_gpu, lda, nullptr, 0, ipiv_gpu, info));

    hip_catch(hipDeviceSynchronize());
#endif

    hip_catch(hipMemcpy(a, a_gpu, a_size * sizeof(math_type), hipMemcpyDeviceToHost));
    hip_catch(hipFree(a_gpu));

    if (ipiv != nullptr) {
        hip_catch(hipMemcpy(ipiv, ipiv_gpu, n * sizeof(int), hipMemcpyDeviceToHost));
        hip_catch(hipFree(ipiv_gpu));
    }

    hip_catch(hipHostUnregister(info));

    int out = *info;

    delete info;

    return out;
}

} // namespace einsums::backend::linear_algebra::hipblas