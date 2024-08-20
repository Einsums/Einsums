//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"

#include "einsums/Exception.hpp"

#include <cstring>
#include <hip/hip_common.h>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
#include <omp.h>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::gpu)

/**
 * @def __host_ptr__
 *
 * This is a macro for decorating pointers to tell the user that a pointer is accessible from the host.
 */
#define __host_ptr__

/**
 * @def __device_ptr__
 *
 * This is a macro for decorating pointers to tell the user that a pointer should be accessible from the device.
 * The pointer can be located anywhere, it just needs to be accessible.
 */
#define __device_ptr__

namespace detail {
/**
 * @struct hip_exception
 *
 * @brief Wraps an hipError_t value as an exception object to be handled by C++'s exception system.
 */
template <hipError_t error>
struct hip_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * @brief Construct a new HIP exception.
     */
    hip_exception(const char *diagnostic) : message{""} {
        message += diagnostic;
        message += hipGetErrorString(error);
    }

    hip_exception(std::string diagnostic) : message{""} {
        message += diagnostic;
        message += hipGetErrorString(error);
    }

    /**
     * @brief Return the error string.
     */
    const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     */
    template <hipError_t other_error>
    bool operator==(const hip_exception<other_error> &other) const {
        return error == other_error;
    }

    /**
     * @brief Equality operator.
     */
    bool operator==(hipError_t other) const { return error == other; }

    /**
     * @brief Inequality operator.
     */
    template <hipError_t other_error>
    bool operator!=(const hip_exception<other_error> &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(hipError_t other) const { return error != other; }

    friend bool operator==(hipError_t, const hip_exception<error> &);
    friend bool operator!=(hipError_t, const hip_exception<error> &);
};

/**
 * @brief Reverse equality operator.
 */
template <hipError_t error>
bool operator==(hipError_t first, const hip_exception<error> &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 */
template <hipError_t error>
bool operator!=(hipError_t first, const hip_exception<error> &second) {
    return first != error;
}

// Auto-generated code.
using Success                          = hip_exception<hipSuccess>;
using ErrorInvalidValue                = hip_exception<hipErrorInvalidValue>;
using ErrorOutOfMemory                 = hip_exception<hipErrorOutOfMemory>;
using ErrorMemoryAllocation            = hip_exception<hipErrorMemoryAllocation>;
using ErrorNotInitialized              = hip_exception<hipErrorNotInitialized>;
using ErrorInitializationError         = hip_exception<hipErrorInitializationError>;
using ErrorDeinitialized               = hip_exception<hipErrorDeinitialized>;
using ErrorProfilerDisabled            = hip_exception<hipErrorProfilerDisabled>;
using ErrorProfilerNotInitialized      = hip_exception<hipErrorProfilerNotInitialized>;
using ErrorProfilerAlreadyStarted      = hip_exception<hipErrorProfilerAlreadyStarted>;
using ErrorProfilerAlreadyStopped      = hip_exception<hipErrorProfilerAlreadyStopped>;
using ErrorInvalidConfiguration        = hip_exception<hipErrorInvalidConfiguration>;
using ErrorInvalidPitchValue           = hip_exception<hipErrorInvalidPitchValue>;
using ErrorInvalidSymbol               = hip_exception<hipErrorInvalidSymbol>;
using ErrorInvalidDevicePointer        = hip_exception<hipErrorInvalidDevicePointer>;
using ErrorInvalidMemcpyDirection      = hip_exception<hipErrorInvalidMemcpyDirection>;
using ErrorInsufficientDriver          = hip_exception<hipErrorInsufficientDriver>;
using ErrorMissingConfiguration        = hip_exception<hipErrorMissingConfiguration>;
using ErrorPriorLaunchFailure          = hip_exception<hipErrorPriorLaunchFailure>;
using ErrorInvalidDeviceFunction       = hip_exception<hipErrorInvalidDeviceFunction>;
using ErrorNoDevice                    = hip_exception<hipErrorNoDevice>;
using ErrorInvalidDevice               = hip_exception<hipErrorInvalidDevice>;
using ErrorInvalidImage                = hip_exception<hipErrorInvalidImage>;
using ErrorInvalidContext              = hip_exception<hipErrorInvalidContext>;
using ErrorContextAlreadyCurrent       = hip_exception<hipErrorContextAlreadyCurrent>;
using ErrorMapFailed                   = hip_exception<hipErrorMapFailed>;
using ErrorMapBufferObjectFailed       = hip_exception<hipErrorMapBufferObjectFailed>;
using ErrorUnmapFailed                 = hip_exception<hipErrorUnmapFailed>;
using ErrorArrayIsMapped               = hip_exception<hipErrorArrayIsMapped>;
using ErrorAlreadyMapped               = hip_exception<hipErrorAlreadyMapped>;
using ErrorNoBinaryForGpu              = hip_exception<hipErrorNoBinaryForGpu>;
using ErrorAlreadyAcquired             = hip_exception<hipErrorAlreadyAcquired>;
using ErrorNotMapped                   = hip_exception<hipErrorNotMapped>;
using ErrorNotMappedAsArray            = hip_exception<hipErrorNotMappedAsArray>;
using ErrorNotMappedAsPointer          = hip_exception<hipErrorNotMappedAsPointer>;
using ErrorECCNotCorrectable           = hip_exception<hipErrorECCNotCorrectable>;
using ErrorUnsupportedLimit            = hip_exception<hipErrorUnsupportedLimit>;
using ErrorContextAlreadyInUse         = hip_exception<hipErrorContextAlreadyInUse>;
using ErrorPeerAccessUnsupported       = hip_exception<hipErrorPeerAccessUnsupported>;
using ErrorInvalidKernelFile           = hip_exception<hipErrorInvalidKernelFile>;
using ErrorInvalidGraphicsContext      = hip_exception<hipErrorInvalidGraphicsContext>;
using ErrorInvalidSource               = hip_exception<hipErrorInvalidSource>;
using ErrorFileNotFound                = hip_exception<hipErrorFileNotFound>;
using ErrorSharedObjectSymbolNotFound  = hip_exception<hipErrorSharedObjectSymbolNotFound>;
using ErrorSharedObjectInitFailed      = hip_exception<hipErrorSharedObjectInitFailed>;
using ErrorOperatingSystem             = hip_exception<hipErrorOperatingSystem>;
using ErrorInvalidHandle               = hip_exception<hipErrorInvalidHandle>;
using ErrorInvalidResourceHandle       = hip_exception<hipErrorInvalidResourceHandle>;
using ErrorIllegalState                = hip_exception<hipErrorIllegalState>;
using ErrorNotFound                    = hip_exception<hipErrorNotFound>;
using ErrorNotReady                    = hip_exception<hipErrorNotReady>;
using ErrorIllegalAddress              = hip_exception<hipErrorIllegalAddress>;
using ErrorLaunchOutOfResources        = hip_exception<hipErrorLaunchOutOfResources>;
using ErrorLaunchTimeOut               = hip_exception<hipErrorLaunchTimeOut>;
using ErrorPeerAccessAlreadyEnabled    = hip_exception<hipErrorPeerAccessAlreadyEnabled>;
using ErrorPeerAccessNotEnabled        = hip_exception<hipErrorPeerAccessNotEnabled>;
using ErrorSetOnActiveProcess          = hip_exception<hipErrorSetOnActiveProcess>;
using ErrorContextIsDestroyed          = hip_exception<hipErrorContextIsDestroyed>;
using ErrorAssert                      = hip_exception<hipErrorAssert>;
using ErrorHostMemoryAlreadyRegistered = hip_exception<hipErrorHostMemoryAlreadyRegistered>;
using ErrorHostMemoryNotRegistered     = hip_exception<hipErrorHostMemoryNotRegistered>;
using ErrorLaunchFailure               = hip_exception<hipErrorLaunchFailure>;
using ErrorCooperativeLaunchTooLarge   = hip_exception<hipErrorCooperativeLaunchTooLarge>;
using ErrorNotSupported                = hip_exception<hipErrorNotSupported>;
using ErrorStreamCaptureUnsupported    = hip_exception<hipErrorStreamCaptureUnsupported>;
using ErrorStreamCaptureInvalidated    = hip_exception<hipErrorStreamCaptureInvalidated>;
using ErrorStreamCaptureMerge          = hip_exception<hipErrorStreamCaptureMerge>;
using ErrorStreamCaptureUnmatched      = hip_exception<hipErrorStreamCaptureUnmatched>;
using ErrorStreamCaptureUnjoined       = hip_exception<hipErrorStreamCaptureUnjoined>;
using ErrorStreamCaptureIsolation      = hip_exception<hipErrorStreamCaptureIsolation>;
using ErrorStreamCaptureImplicit       = hip_exception<hipErrorStreamCaptureImplicit>;
using ErrorCapturedEvent               = hip_exception<hipErrorCapturedEvent>;
using ErrorStreamCaptureWrongThread    = hip_exception<hipErrorStreamCaptureWrongThread>;
using ErrorGraphExecUpdateFailure      = hip_exception<hipErrorGraphExecUpdateFailure>;
using ErrorUnknown                     = hip_exception<hipErrorUnknown>;
using ErrorRuntimeMemory               = hip_exception<hipErrorRuntimeMemory>;
using ErrorRuntimeOther                = hip_exception<hipErrorRuntimeOther>;
using ErrorTbd                         = hip_exception<hipErrorTbd>;

/**
 * @struct hipblas_exception
 * @brief Wraps hipBLAS status codes into an exception.
 *
 * Wraps hipBLAS status codes so that they can be thrown and caught. There is one code for each named hipBLAS status code.
 *
 * @tparam error The status code handled by this exception.
 */
template <hipblasStatus_t error>
struct hipblas_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * @brief Construct a new hipblas_exception.
     */
    hipblas_exception(const char *diagnostic) : message{""} {
        message += diagnostic;
        message += hipblasStatusToString(error);
    }

    hipblas_exception(std::string diagnostic) : message{""} {
        message += diagnostic;
        message += hipblasStatusToString(error);
    }

    /**
     * @brief Return the error string corresponding to the status code.
     */
    const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     */
    template <hipblasStatus_t other_error>
    bool operator==(const hipblas_exception<other_error> &other) const {
        return error == other_error;
    }

    /**
     * @brief Equality operator.
     */
    bool operator==(hipblasStatus_t other) const { return error == other; }

    /**
     * @brief Inequality operator.
     */
    template <hipblasStatus_t other_error>
    bool operator!=(const hipblas_exception<other_error> &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(hipblasStatus_t other) const { return error != other; }

    friend bool operator==(hipblasStatus_t, const hipblas_exception<error> &);
    friend bool operator!=(hipblasStatus_t, const hipblas_exception<error> &);
};

/**
 * @brief Reverse equality operator.
 */
template <hipblasStatus_t error>
bool operator==(hipblasStatus_t first, const hipblas_exception<error> &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 */
template <hipblasStatus_t error>
bool operator!=(hipblasStatus_t first, const hipblas_exception<error> &second) {
    return first != error;
}

// Put the status code documentaion in a different header for cleaner code.
#include "einsums/gpu/hipblas_status_doc.hpp"

using blasSuccess         = hipblas_exception<HIPBLAS_STATUS_SUCCESS>;
using blasNotInitialized  = hipblas_exception<HIPBLAS_STATUS_NOT_INITIALIZED>;
using blasAllocFailed     = hipblas_exception<HIPBLAS_STATUS_ALLOC_FAILED>;
using blasInvalidValue    = hipblas_exception<HIPBLAS_STATUS_INVALID_VALUE>;
using blasMappingError    = hipblas_exception<HIPBLAS_STATUS_MAPPING_ERROR>;
using blasExecutionFailed = hipblas_exception<HIPBLAS_STATUS_EXECUTION_FAILED>;
using blasInternalError   = hipblas_exception<HIPBLAS_STATUS_INTERNAL_ERROR>;
using blasNotSupported    = hipblas_exception<HIPBLAS_STATUS_NOT_SUPPORTED>;
using blasArchMismatch    = hipblas_exception<HIPBLAS_STATUS_ARCH_MISMATCH>;
using blasHandleIsNullptr = hipblas_exception<HIPBLAS_STATUS_HANDLE_IS_NULLPTR>;
using blasInvalidEnum     = hipblas_exception<HIPBLAS_STATUS_INVALID_ENUM>;
using blasUnknown         = hipblas_exception<HIPBLAS_STATUS_UNKNOWN>;

/**
 * @brief Create a string representation of an hipsolverStatus_t value.
 * Create a string representation of an hipsolverStatus_t value. There is no
 * equivalent function in hipSolver at this point in time, so a custom one
 * had to be made.
 *
 * @param status The status code to convert.
 *
 * @return A pointer to a string containing a brief message detailing the status.
 */
EINSUMS_EXPORT const char *hipsolverStatusToString(hipsolverStatus_t status);

/**
 * @struct hipsolver_exception
 *
 * @brief Wraps hipSolver status codes into an exception.
 *
 * Wraps hipSolver status codes into an exception which allows them to be thrown and caught.
 *
 * @tparam error The status code wrapped by the object.
 */
template <hipsolverStatus_t error>
struct hipsolver_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * Construct a new exception.
     */
    hipsolver_exception(const char *diagnostic) : message{""} {
        message += diagnostic;
        message += hipsolverStatusToString(error);
    }

    hipsolver_exception(std::string diagnostic) : message{""} {
        message += diagnostic;
        message += hipsolverStatusToString(error);
    }

    /**
     * Return the error string.
     */
    const char *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * Equality operator.
     */
    template <hipsolverStatus_t other_error>
    bool operator==(const hipsolver_exception<other_error> &other) const {
        return error == other_error;
    }

    /**
     * Equality operator.
     */
    bool operator==(hipsolverStatus_t other) const { return error == other; }

    /**
     * Inquality operator.
     */
    template <hipsolverStatus_t other_error>
    bool operator!=(const hipsolver_exception<other_error> &other) const {
        return error != other_error;
    }

    /**
     * Inquality operator.
     */
    bool operator!=(hipsolverStatus_t other) const { return error != other; }

    friend bool operator==(hipsolverStatus_t, const hipsolver_exception<error> &);
    friend bool operator!=(hipsolverStatus_t, const hipsolver_exception<error> &);
};

/**
 * Reverse equality operator.
 */
template <hipsolverStatus_t error>
bool operator==(hipsolverStatus_t first, const hipsolver_exception<error> &second) {
    return first == error;
}

/**
 * Reverse inequality operator.
 */
template <hipsolverStatus_t error>
bool operator!=(hipsolverStatus_t first, const hipsolver_exception<error> &second) {
    return first != error;
}

using solverSuccess          = hipsolver_exception<HIPSOLVER_STATUS_SUCCESS>;
using solverNotInitialized   = hipsolver_exception<HIPSOLVER_STATUS_NOT_INITIALIZED>;
using solverAllocFailed      = hipsolver_exception<HIPSOLVER_STATUS_ALLOC_FAILED>;
using solverInvalidValue     = hipsolver_exception<HIPSOLVER_STATUS_INVALID_VALUE>;
using solverMappingError     = hipsolver_exception<HIPSOLVER_STATUS_MAPPING_ERROR>;
using solverExecutionFailed  = hipsolver_exception<HIPSOLVER_STATUS_EXECUTION_FAILED>;
using solverInternalError    = hipsolver_exception<HIPSOLVER_STATUS_INTERNAL_ERROR>;
using solverFuncNotSupported = hipsolver_exception<HIPSOLVER_STATUS_NOT_SUPPORTED>;
using solverArchMismatch     = hipsolver_exception<HIPSOLVER_STATUS_ARCH_MISMATCH>;
using solverHandleIsNullptr  = hipsolver_exception<HIPSOLVER_STATUS_HANDLE_IS_NULLPTR>;
using solverInvalidEnum      = hipsolver_exception<HIPSOLVER_STATUS_INVALID_ENUM>;
using solverUnknown          = hipsolver_exception<HIPSOLVER_STATUS_UNKNOWN>;
// using solverZeroPivot = hipsolver_exception<HIPSOLVER_STATUS_ZERO_PIVOT>;
} // namespace detail

// Get the handles used internally. Can be used by other blas and solver processes.
/**
 * @brief Get the internal hipBLAS handle.
 *
 * @return The current internal hipBLAS handle.
 */
EINSUMS_EXPORT hipblasHandle_t get_blas_handle();
EINSUMS_EXPORT hipblasHandle_t get_blas_handle(int thread_id);

/**
 * @brief Get the internal hipSolver handle.
 *
 * @return The current internal hipSolver handle.
 */
EINSUMS_EXPORT hipsolverHandle_t get_solver_handle();
EINSUMS_EXPORT hipsolverHandle_t get_solver_handle(int thread_id);

// Set the handles used internally. Useful to avoid creating multiple contexts.
/**
 * @brief Set the internal hipBLAS handle.
 *
 * @param value The new handle.
 *
 * @return The new handle.
 */
EINSUMS_EXPORT hipblasHandle_t set_blas_handle(hipblasHandle_t value);
EINSUMS_EXPORT hipblasHandle_t set_blas_handle(hipblasHandle_t value, int thread_id);

/**
 * @brief Set the internal hipSolver handle.
 *
 * @param value The new handle.
 *
 * @return The new handle.
 */
EINSUMS_EXPORT hipsolverHandle_t set_solver_handle(hipsolverHandle_t value);
EINSUMS_EXPORT hipsolverHandle_t set_solver_handle(hipsolverHandle_t value, int thread_id);

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 */
__host__ EINSUMS_EXPORT void __hipblas_catch__(hipblasStatus_t status, const char *func_call, const char *fname, const char *diagnostic,
                                               const char *funcname, bool throw_success = false);

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 */
__host__ EINSUMS_EXPORT void __hipsolver_catch__(hipsolverStatus_t status, const char *func_call, const char *fname, const char *diagnostic,
                                                 const char *funcname, bool throw_success = false);

/**
 * Wraps up an HIP function to catch any error codes. If the function does not return
 * hipSuccess, then an exception will be thrown
 */
__host__ EINSUMS_EXPORT void __hip_catch__(hipError_t condition, const char *func_call, const char *fname, const char *diagnostic,
                                           const char *funcname, bool throw_success = false);

#define hip_catch_STR1(x) #x
#define hip_catch_STR(x)  hip_catch_STR1(x)
#define hip_catch(condition, ...)                                                                                                          \
    __hip_catch__(                                                                                                                         \
        (condition), "hip_catch" hip_catch_STR((condition)) ";", einsums::detail::anonymize(__FILE__).c_str(),                             \
        ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

#define hipblas_catch(condition, ...)                                                                                                      \
    __hipblas_catch__(                                                                                                                     \
        (condition), "hipblas_catch" hip_catch_STR((condition)) ";", einsums::detail::anonymize(__FILE__).c_str(),                         \
        ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

#define hipsolver_catch(condition, ...)                                                                                                    \
    __hipsolver_catch__(                                                                                                                   \
        (condition), "hipsolver_catch" hip_catch_STR((condition)) ";", einsums::detail::anonymize(__FILE__).c_str(),                       \
        ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

/**
 * @def get_worker_info
 * @brief Get the worker thread launch parameters on the GPU.
 */
#define get_worker_info(thread_id, num_threads)                                                                                            \
    num_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;                                                \
    thread_id   = threadIdx.x +                                                                                                            \
                blockDim.x * (threadIdx.y +                                                                                                \
                              blockDim.y * (threadIdx.z + blockDim.z * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))));

/**
 * Initialize the GPU and HIP.
 */
__host__ EINSUMS_EXPORT void initialize();

/**
 * Finalize HIP.
 */
__host__ EINSUMS_EXPORT void finalize();

/**
 * Returns the memory pool used for allocating scale values in linear algebra calls.
 * This should hopefully fix the issue of allocating a whole page of memory for a single float.
 */
__host__ EINSUMS_EXPORT hipMemPool_t &get_scale_pool();

#define KERNEL(bound) __global__ __launch_bounds__((bound))

/**
 * Get an appropriate block size for a kernel.
 */
__host__ inline dim3 block_size(size_t compute_size) {
    if (compute_size < 32) {
        return dim3(32);
    } else if (compute_size < 256) {
        return dim3(compute_size & 0xf0);
    } else {
        return dim3(256);
    }
}

/**
 * Get an appropriate number of blocks for a kernel.
 */
__host__ inline dim3 blocks(size_t compute_size) {
    if (compute_size < 256) {
        return dim3(1);
    } else if (compute_size <= 4096) {
        return dim3(compute_size / 256);
    } else {
        return dim3(16);
    }
}

/**
 * @brief Wait on a stream.
 */
__host__ EINSUMS_EXPORT void stream_wait(hipStream_t stream);

/**
 * @brief Indicates that the next skippable wait on the current thread should be skipped.
 * Does not apply to stream_wait(stream) with the stream specified, all_stream_wait, or device_synchronize.
 * Does not affect stream_wait(false), and stream_wait(false) does not affect the skip state.
 */
__host__ EINSUMS_EXPORT void skip_next_wait();

/**
 * @brief Wait on the current thread's stream. Can be skippable or not.
 *
 * @param may_skip Indicate that the wait may be skipped to avoid unnecessary waits. Only skipped after a call to skip_next_wait,
 * then resets the skip flag so that later waits are not skipped.
 */
__host__ EINSUMS_EXPORT void stream_wait(bool may_skip = true);

/**
 * @brief Wait on all streams managed by Einsums.
 */
__host__ EINSUMS_EXPORT void all_stream_wait();

// Because I want it to be plural as well
#define all_streams_wait() all_stream_wait()

/**
 * @brief Gets the stream assigned to the current thread.
 */
EINSUMS_EXPORT hipStream_t get_stream();
EINSUMS_EXPORT hipStream_t get_stream(int thread_id);

/**
 * @brief Sets the stream assigned to the current thread.
 */
EINSUMS_EXPORT void set_stream(hipStream_t stream);
EINSUMS_EXPORT void set_stream(hipStream_t stream, int thread_id);

/**
 * @brief Synchronize to all operations on the device.
 *
 * Waits until the device is in an idle state before continuing. Blocks on all streams.
 */
inline void device_synchronize() {
    hip_catch(hipDeviceSynchronize());
}

__device__ inline bool is_zero(double value) {
    return value == 0.0;
}

__device__ inline bool is_zero(float value) {
    return value == 0.0f;
}

__device__ inline bool is_zero(hipComplex value) {
    return value.x == 0.0f && value.y == 0.0f;
}

__device__ inline bool is_zero(hipDoubleComplex value) {
    return value.x == 0.0 && value.y == 0.0;
}

__device__ inline void make_zero(double &value) {
    value = 0.0;
}

__device__ inline void make_zero(float &value) {
    value = 0.0f;
}

__device__ inline void make_zero(hipComplex &value) {
    value.x = 0.0f;
    value.y = 0.0f;
}

__device__ inline void make_zero(hipDoubleComplex &value) {
    value.x = 0.0;
    value.y = 0.0;
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(float *address, float value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(double *address, double value) {
    atomicAdd(address, value);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(hipComplex *address, hipComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

/**
 * @brief Wrap the atomicAdd operation to allow polymorphism on complex arguments.
 */
__device__ inline void atomicAdd_wrap(hipDoubleComplex *address, hipDoubleComplex value) {
    atomicAdd(&(address->x), value.x);
    atomicAdd(&(address->y), value.y);
}

__host__ EINSUMS_EXPORT int get_warpsize(void);

END_EINSUMS_NAMESPACE_HPP(einsums::gpu)