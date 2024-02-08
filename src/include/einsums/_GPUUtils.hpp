//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include "einsums/_Common.hpp"

#include <cstring>
#include <hip/amd_detail/amd_hip_complex.h>
#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
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
  private :
    ::std::string message;

  public:
    /**
     * @brief Construct a new HIP exception.
     */
    hip_exception(const char *diagnostic) : message{""} {
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

} // namespace detail

/**
 * Wraps up an HIP function to catch any error codes. If the function does not return
 * hipSuccess, then an exception will be thrown
 */
__host__ EINSUMS_EXPORT void __hip_catch__(hipError_t condition, const char *diagnostic, bool throw_success = false);

#define hip_catch_STR1(x) #x
#define hip_catch_STR(x) hip_catch_STR1(x)
#define hip_catch(condition, ...) __hip_catch__((condition), __FILE__ ":" hip_catch_STR(__LINE__) ": " __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief Get the worker thread launch parameters on the GPU.
 */
__device__
inline void get_worker_info(int &thread_id, int &num_threads) {
    num_threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    thread_id = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (threadIdx.z + blockDim.z * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z))));
}

__host__
EINSUMS_EXPORT void initialize();

__host__
EINSUMS_EXPORT void finalize();

#define KERNEL(bound) __global__ __launch_bounds__((bound))

END_EINSUMS_NAMESPACE_HPP(einsums::gpu)