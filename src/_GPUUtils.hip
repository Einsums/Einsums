//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::gpu)

__host__ void __hip_catch__(hipError_t condition, const char *diagnostic, bool throw_success) {
    // Auto-generated code.
    switch (condition) {
    case hipSuccess:
        if (throw_success) {
            throw detail::Success(diagnostic);
        } else {
            return;
        }
        break;
    case hipErrorInvalidValue:
        throw detail::ErrorInvalidValue(diagnostic);
        break;
    case hipErrorOutOfMemory:
        throw detail::ErrorOutOfMemory(diagnostic);
        break;
    case hipErrorNotInitialized:
        throw detail::ErrorNotInitialized(diagnostic);
        break;
    case hipErrorDeinitialized:
        throw detail::ErrorDeinitialized(diagnostic);
        break;
    case hipErrorProfilerDisabled:
        throw detail::ErrorProfilerDisabled(diagnostic);
        break;
    case hipErrorProfilerNotInitialized:
        throw detail::ErrorProfilerNotInitialized(diagnostic);
        break;
    case hipErrorProfilerAlreadyStarted:
        throw detail::ErrorProfilerAlreadyStarted(diagnostic);
        break;
    case hipErrorProfilerAlreadyStopped:
        throw detail::ErrorProfilerAlreadyStopped(diagnostic);
        break;
    case hipErrorInvalidConfiguration:
        throw detail::ErrorInvalidConfiguration(diagnostic);
        break;
    case hipErrorInvalidPitchValue:
        throw detail::ErrorInvalidPitchValue(diagnostic);
        break;
    case hipErrorInvalidSymbol:
        throw detail::ErrorInvalidSymbol(diagnostic);
        break;
    case hipErrorInvalidDevicePointer:
        throw detail::ErrorInvalidDevicePointer(diagnostic);
        break;
    case hipErrorInvalidMemcpyDirection:
        throw detail::ErrorInvalidMemcpyDirection(diagnostic);
        break;
    case hipErrorInsufficientDriver:
        throw detail::ErrorInsufficientDriver(diagnostic);
        break;
    case hipErrorMissingConfiguration:
        throw detail::ErrorMissingConfiguration(diagnostic);
        break;
    case hipErrorPriorLaunchFailure:
        throw detail::ErrorPriorLaunchFailure(diagnostic);
        break;
    case hipErrorInvalidDeviceFunction:
        throw detail::ErrorInvalidDeviceFunction(diagnostic);
        break;
    case hipErrorNoDevice:
        throw detail::ErrorNoDevice(diagnostic);
        break;
    case hipErrorInvalidDevice:
        throw detail::ErrorInvalidDevice(diagnostic);
        break;
    case hipErrorInvalidImage:
        throw detail::ErrorInvalidImage(diagnostic);
        break;
    case hipErrorInvalidContext:
        throw detail::ErrorInvalidContext(diagnostic);
        break;
    case hipErrorContextAlreadyCurrent:
        throw detail::ErrorContextAlreadyCurrent(diagnostic);
        break;
    case hipErrorMapFailed:
        throw detail::ErrorMapFailed(diagnostic);
        break;
    case hipErrorUnmapFailed:
        throw detail::ErrorUnmapFailed(diagnostic);
        break;
    case hipErrorArrayIsMapped:
        throw detail::ErrorArrayIsMapped(diagnostic);
        break;
    case hipErrorAlreadyMapped:
        throw detail::ErrorAlreadyMapped(diagnostic);
        break;
    case hipErrorNoBinaryForGpu:
        throw detail::ErrorNoBinaryForGpu(diagnostic);
        break;
    case hipErrorAlreadyAcquired:
        throw detail::ErrorAlreadyAcquired(diagnostic);
        break;
    case hipErrorNotMapped:
        throw detail::ErrorNotMapped(diagnostic);
        break;
    case hipErrorNotMappedAsArray:
        throw detail::ErrorNotMappedAsArray(diagnostic);
        break;
    case hipErrorNotMappedAsPointer:
        throw detail::ErrorNotMappedAsPointer(diagnostic);
        break;
    case hipErrorECCNotCorrectable:
        throw detail::ErrorECCNotCorrectable(diagnostic);
        break;
    case hipErrorUnsupportedLimit:
        throw detail::ErrorUnsupportedLimit(diagnostic);
        break;
    case hipErrorContextAlreadyInUse:
        throw detail::ErrorContextAlreadyInUse(diagnostic);
        break;
    case hipErrorPeerAccessUnsupported:
        throw detail::ErrorPeerAccessUnsupported(diagnostic);
        break;
    case hipErrorInvalidKernelFile:
        throw detail::ErrorInvalidKernelFile(diagnostic);
        break;
    case hipErrorInvalidGraphicsContext:
        throw detail::ErrorInvalidGraphicsContext(diagnostic);
        break;
    case hipErrorInvalidSource:
        throw detail::ErrorInvalidSource(diagnostic);
        break;
    case hipErrorFileNotFound:
        throw detail::ErrorFileNotFound(diagnostic);
        break;
    case hipErrorSharedObjectSymbolNotFound:
        throw detail::ErrorSharedObjectSymbolNotFound(diagnostic);
        break;
    case hipErrorSharedObjectInitFailed:
        throw detail::ErrorSharedObjectInitFailed(diagnostic);
        break;
    case hipErrorOperatingSystem:
        throw detail::ErrorOperatingSystem(diagnostic);
        break;
    case hipErrorInvalidHandle:
        throw detail::ErrorInvalidHandle(diagnostic);
        break;
    case hipErrorIllegalState:
        throw detail::ErrorIllegalState(diagnostic);
        break;
    case hipErrorNotFound:
        throw detail::ErrorNotFound(diagnostic);
        break;
    case hipErrorNotReady:
        throw detail::ErrorNotReady(diagnostic);
        break;
    case hipErrorIllegalAddress:
        throw detail::ErrorIllegalAddress(diagnostic);
        break;
    case hipErrorLaunchOutOfResources:
        throw detail::ErrorLaunchOutOfResources(diagnostic);
        break;
    case hipErrorLaunchTimeOut:
        throw detail::ErrorLaunchTimeOut(diagnostic);
        break;
    case hipErrorPeerAccessAlreadyEnabled:
        throw detail::ErrorPeerAccessAlreadyEnabled(diagnostic);
        break;
    case hipErrorPeerAccessNotEnabled:
        throw detail::ErrorPeerAccessNotEnabled(diagnostic);
        break;
    case hipErrorSetOnActiveProcess:
        throw detail::ErrorSetOnActiveProcess(diagnostic);
        break;
    case hipErrorContextIsDestroyed:
        throw detail::ErrorContextIsDestroyed(diagnostic);
        break;
    case hipErrorAssert:
        throw detail::ErrorAssert(diagnostic);
        break;
    case hipErrorHostMemoryAlreadyRegistered:
        throw detail::ErrorHostMemoryAlreadyRegistered(diagnostic);
        break;
    case hipErrorHostMemoryNotRegistered:
        throw detail::ErrorHostMemoryNotRegistered(diagnostic);
        break;
    case hipErrorLaunchFailure:
        throw detail::ErrorLaunchFailure(diagnostic);
        break;
    case hipErrorCooperativeLaunchTooLarge:
        throw detail::ErrorCooperativeLaunchTooLarge(diagnostic);
        break;
    case hipErrorNotSupported:
        throw detail::ErrorNotSupported(diagnostic);
        break;
    case hipErrorStreamCaptureUnsupported:
        throw detail::ErrorStreamCaptureUnsupported(diagnostic);
        break;
    case hipErrorStreamCaptureInvalidated:
        throw detail::ErrorStreamCaptureInvalidated(diagnostic);
        break;
    case hipErrorStreamCaptureMerge:
        throw detail::ErrorStreamCaptureMerge(diagnostic);
        break;
    case hipErrorStreamCaptureUnmatched:
        throw detail::ErrorStreamCaptureUnmatched(diagnostic);
        break;
    case hipErrorStreamCaptureUnjoined:
        throw detail::ErrorStreamCaptureUnjoined(diagnostic);
        break;
    case hipErrorStreamCaptureIsolation:
        throw detail::ErrorStreamCaptureIsolation(diagnostic);
        break;
    case hipErrorStreamCaptureImplicit:
        throw detail::ErrorStreamCaptureImplicit(diagnostic);
        break;
    case hipErrorCapturedEvent:
        throw detail::ErrorCapturedEvent(diagnostic);
        break;
    case hipErrorStreamCaptureWrongThread:
        throw detail::ErrorStreamCaptureWrongThread(diagnostic);
        break;
    case hipErrorGraphExecUpdateFailure:
        throw detail::ErrorGraphExecUpdateFailure(diagnostic);
        break;
    case hipErrorUnknown:
        throw detail::ErrorUnknown(diagnostic);
        break;
    case hipErrorRuntimeMemory:
        throw detail::ErrorRuntimeMemory(diagnostic);
        break;
    case hipErrorRuntimeOther:
        throw detail::ErrorRuntimeOther(diagnostic);
        break;
    case hipErrorTbd:
        throw detail::ErrorTbd(diagnostic);
        break;
    default:
        throw detail::ErrorUnknown(diagnostic);
    }
}

namespace detail {
std::vector<hipblasHandle_t> *blas_handles = nullptr;
std::vector<hipsolverHandle_t> *solver_handles = nullptr;
std::vector<hipStream_t> *default_streams = nullptr;
}

__host__ void __hipblas_catch__(hipblasStatus_t status, const char *diagnostic, bool throw_success) {
    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        if (throw_success) {
            throw detail::blasSuccess(diagnostic);
        } else {
            return;
        }
        break;
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        throw detail::blasNotInitialized(diagnostic);
        break;
    case HIPBLAS_STATUS_ALLOC_FAILED:
        throw detail::blasAllocFailed(diagnostic);
        break;
    case HIPBLAS_STATUS_INVALID_VALUE:
        throw detail::blasInvalidValue(diagnostic);
        break;
    case HIPBLAS_STATUS_MAPPING_ERROR:
        throw detail::blasMappingError(diagnostic);
        break;
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        throw detail::blasExecutionFailed(diagnostic);
        break;
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        throw detail::blasInternalError(diagnostic);
        break;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        throw detail::blasNotSupported(diagnostic);
        break;
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        throw detail::blasArchMismatch(diagnostic);
        break;
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
        throw detail::blasHandleIsNullptr(diagnostic);
        break;
    case HIPBLAS_STATUS_INVALID_ENUM:
        throw detail::blasInvalidEnum(diagnostic);
        break;
    default:
        throw detail::blasUnknown(diagnostic);
    }
}

hipblasHandle_t get_blas_handle(int thread_id) {
    return detail::blas_handles->at(thread_id);
}

hipblasHandle_t get_blas_handle() {
    return get_blas_handle(omp_get_thread_num());
}

hipblasHandle_t set_blas_handle(hipblasHandle_t value, int thread_id) {
    detail::blas_handles->at(thread_id) = value;
    return detail::blas_handles->at(thread_id);
}

hipblasHandle_t set_blas_handle(hipblasHandle_t value) {
    return set_blas_handle(value, omp_get_thread_num());
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
namespace detail {
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

}

hipsolverHandle_t get_solver_handle(int thread_id) {
    return detail::solver_handles->at(thread_id);
}

hipsolverHandle_t get_solver_handle() {
    return get_solver_handle(omp_get_thread_num());
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value, int thread_id) {
    detail::solver_handles->at(thread_id) = value;
    return detail::solver_handles->at(thread_id);
}

hipsolverHandle_t set_solver_handle(hipsolverHandle_t value) {
    return set_solver_handle(value, omp_get_thread_num());
}

static hipMemPool_t pool;

void initialize() {
    hip_catch(hipFree(nullptr));

    hip_catch(hipSetDevice(0));

    hipMemPoolProps props;

    props.allocType = hipMemAllocationTypePinned;
    props.handleTypes = (hipMemAllocationHandleType) (hipMemHandleTypePosixFileDescriptor | hipMemHandleTypeWin32Kmt);
    props.location = {hipMemLocationTypeDevice, 0};


    hip_catch(hipMemPoolCreate(&pool, &props));

    if(detail::default_streams == nullptr) {
        detail::default_streams = new std::vector<hipStream_t>(omp_get_num_threads());

        for(int i = 0; i < omp_get_num_threads(); i++) {
            hip_catch(hipStreamCreateWithFlags(&(detail::default_streams->at(i)), hipStreamNonBlocking));
        }
    }

    if (detail::blas_handles == nullptr) {
        detail::blas_handles = new std::vector<hipblasHandle_t>(omp_get_num_threads());

        for(int i = 0; i < omp_get_num_threads(); i++) {
            hipblas_catch(hipblasCreate(&(detail::blas_handles->at(i))));
            hipblas_catch(hipblasSetStream(detail::blas_handles->at(i), detail::default_streams->at(i)));
        }
    }

    if (detail::solver_handles == nullptr) {
        detail::solver_handles = new std::vector<hipsolverHandle_t>(omp_get_num_threads());

        for(int i = 0; i < omp_get_num_threads(); i++) {
            hipsolver_catch(hipsolverCreate(&(detail::solver_handles->at(i))));
            hipsolver_catch(hipsolverSetStream(detail::solver_handles->at(i), detail::default_streams->at(i)));
        }
    }
}

void finalize() {
    if (detail::blas_handles != nullptr) {
        for(int i = 0; i < omp_get_num_threads(); i++) {
            hipblas_catch(hipblasDestroy(detail::blas_handles->at(i)));
        }
        delete detail::blas_handles;
        detail::blas_handles = nullptr;
    }

    if (detail::solver_handles != nullptr) {
        for(int i = 0; i < omp_get_num_threads(); i++) {
            hipsolver_catch(hipsolverDestroy(detail::solver_handles->at(i)));
        }
        delete detail::solver_handles;
        detail::solver_handles = nullptr;
    }

    if(detail::default_streams != nullptr) {
        for(int i = 0; i < detail::default_streams->size(); i++) {
            hip_catch(hipStreamDestroy(detail::default_streams->at(i)));
        }
        delete detail::default_streams;
        detail::default_streams = nullptr;
    }
    // hip_catch(hipDeviceReset());
    hip_catch(hipMemPoolDestroy(pool));
}

__host__ hipMemPool_t &get_scale_pool() {
    return pool;
}

__host__
void stream_wait() {
    stream_wait(get_stream());
}

__host__
void stream_wait(hipStream_t stream) {
    hipEvent_t wait_event;

    hip_catch(hipEventCreate(&wait_event));
    hip_catch(hipEventRecord(wait_event, stream));

    hip_catch(hipEventSynchronize(wait_event));
    hip_catch(hipEventDestroy(wait_event));
}

__host__
void all_stream_wait() {

    std::vector<hipEvent_t> wait_events(omp_get_num_threads());

    for(int i = 0; i < wait_events.size(); i++) {
        hip_catch(hipEventCreate(&(wait_events.at(i))));
        hip_catch(hipEventRecord(wait_events.at(i), detail::default_streams->at(i)));
    }

    for(int i = 0; i < wait_events.size(); i++) {
        hip_catch(hipEventSynchronize(wait_events.at(i)));
        hip_catch(hipEventDestroy(wait_events.at(i)));
    }
}

/**
 * @brief Gets the stream assigned to the current thread.
 */
hipStream_t get_stream() {
    return get_stream(omp_get_thread_num());
}
hipStream_t get_stream(int thread_id) {
    if(detail::default_streams == nullptr) {
        return nullptr;
    }
    return detail::default_streams->at(thread_id);
}

/**
 * @brief Sets the stream assigned to the current thread.
 */
void set_stream(hipStream_t stream) {
    set_stream(stream, omp_get_thread_num());
}

void set_stream(hipStream_t stream, int thread_id) {
    detail::default_streams->at(thread_id) = stream;

    hipblas_catch(hipblasSetStream(get_blas_handle(), stream));
    hipsolver_catch(hipsolverSetStream(get_solver_handle(), stream));
}

END_EINSUMS_NAMESPACE_CPP(einsums::gpu)