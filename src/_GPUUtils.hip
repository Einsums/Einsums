//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

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

static hipMemPool_t pool;

void initialize() {
    hip_catch(hipFree(nullptr));

    hip_catch(hipSetDevice(0));

    hipMemPoolProps props;

    props.allocType = hipMemAllocationTypePinned;
    props.handleTypes = (hipMemAllocationHandleType) (hipMemHandleTypePosixFileDescriptor | hipMemHandleTypeWin32Kmt);
    props.location = {hipMemLocationTypeDevice, 0};


    hip_catch(hipMemPoolCreate(&pool, &props));
}

void finalize() {
    // hip_catch(hipDeviceReset());
    hip_catch(hipMemPoolDestroy(pool));
}

__host__ hipMemPool_t &get_scale_pool() {
    return pool;
}

END_EINSUMS_NAMESPACE_CPP(einsums::gpu)