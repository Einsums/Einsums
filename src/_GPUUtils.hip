//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <hip/hip_common.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "einsums/_Common.hpp"
#include "einsums/_GPUUtils.hpp"

BEGIN_EINSUMS_NAMESPACE_CPP(einsums::gpu)

__host__ void hip_catch(hipError_t condition, bool throw_success) {
    // Auto-generated code.
    switch (condition) {
    case hipSuccess:
        if(throw_success) {
            throw detail::Success();
        } else {
            return;
        }
        break;
    case hipErrorInvalidValue:
        throw detail::ErrorInvalidValue();
        break;
    case hipErrorOutOfMemory:
        throw detail::ErrorOutOfMemory();
        break;
    case hipErrorNotInitialized:
        throw detail::ErrorNotInitialized();
        break;
    case hipErrorDeinitialized:
        throw detail::ErrorDeinitialized();
        break;
    case hipErrorProfilerDisabled:
        throw detail::ErrorProfilerDisabled();
        break;
    case hipErrorProfilerNotInitialized:
        throw detail::ErrorProfilerNotInitialized();
        break;
    case hipErrorProfilerAlreadyStarted:
        throw detail::ErrorProfilerAlreadyStarted();
        break;
    case hipErrorProfilerAlreadyStopped:
        throw detail::ErrorProfilerAlreadyStopped();
        break;
    case hipErrorInvalidConfiguration:
        throw detail::ErrorInvalidConfiguration();
        break;
    case hipErrorInvalidPitchValue:
        throw detail::ErrorInvalidPitchValue();
        break;
    case hipErrorInvalidSymbol:
        throw detail::ErrorInvalidSymbol();
        break;
    case hipErrorInvalidDevicePointer:
        throw detail::ErrorInvalidDevicePointer();
        break;
    case hipErrorInvalidMemcpyDirection:
        throw detail::ErrorInvalidMemcpyDirection();
        break;
    case hipErrorInsufficientDriver:
        throw detail::ErrorInsufficientDriver();
        break;
    case hipErrorMissingConfiguration:
        throw detail::ErrorMissingConfiguration();
        break;
    case hipErrorPriorLaunchFailure:
        throw detail::ErrorPriorLaunchFailure();
        break;
    case hipErrorInvalidDeviceFunction:
        throw detail::ErrorInvalidDeviceFunction();
        break;
    case hipErrorNoDevice:
        throw detail::ErrorNoDevice();
        break;
    case hipErrorInvalidDevice:
        throw detail::ErrorInvalidDevice();
        break;
    case hipErrorInvalidImage:
        throw detail::ErrorInvalidImage();
        break;
    case hipErrorInvalidContext:
        throw detail::ErrorInvalidContext();
        break;
    case hipErrorContextAlreadyCurrent:
        throw detail::ErrorContextAlreadyCurrent();
        break;
    case hipErrorMapFailed:
        throw detail::ErrorMapFailed();
        break;
    case hipErrorUnmapFailed:
        throw detail::ErrorUnmapFailed();
        break;
    case hipErrorArrayIsMapped:
        throw detail::ErrorArrayIsMapped();
        break;
    case hipErrorAlreadyMapped:
        throw detail::ErrorAlreadyMapped();
        break;
    case hipErrorNoBinaryForGpu:
        throw detail::ErrorNoBinaryForGpu();
        break;
    case hipErrorAlreadyAcquired:
        throw detail::ErrorAlreadyAcquired();
        break;
    case hipErrorNotMapped:
        throw detail::ErrorNotMapped();
        break;
    case hipErrorNotMappedAsArray:
        throw detail::ErrorNotMappedAsArray();
        break;
    case hipErrorNotMappedAsPointer:
        throw detail::ErrorNotMappedAsPointer();
        break;
    case hipErrorECCNotCorrectable:
        throw detail::ErrorECCNotCorrectable();
        break;
    case hipErrorUnsupportedLimit:
        throw detail::ErrorUnsupportedLimit();
        break;
    case hipErrorContextAlreadyInUse:
        throw detail::ErrorContextAlreadyInUse();
        break;
    case hipErrorPeerAccessUnsupported:
        throw detail::ErrorPeerAccessUnsupported();
        break;
    case hipErrorInvalidKernelFile:
        throw detail::ErrorInvalidKernelFile();
        break;
    case hipErrorInvalidGraphicsContext:
        throw detail::ErrorInvalidGraphicsContext();
        break;
    case hipErrorInvalidSource:
        throw detail::ErrorInvalidSource();
        break;
    case hipErrorFileNotFound:
        throw detail::ErrorFileNotFound();
        break;
    case hipErrorSharedObjectSymbolNotFound:
        throw detail::ErrorSharedObjectSymbolNotFound();
        break;
    case hipErrorSharedObjectInitFailed:
        throw detail::ErrorSharedObjectInitFailed();
        break;
    case hipErrorOperatingSystem:
        throw detail::ErrorOperatingSystem();
        break;
    case hipErrorInvalidHandle:
        throw detail::ErrorInvalidHandle();
        break;
    case hipErrorIllegalState:
        throw detail::ErrorIllegalState();
        break;
    case hipErrorNotFound:
        throw detail::ErrorNotFound();
        break;
    case hipErrorNotReady:
        throw detail::ErrorNotReady();
        break;
    case hipErrorIllegalAddress:
        throw detail::ErrorIllegalAddress();
        break;
    case hipErrorLaunchOutOfResources:
        throw detail::ErrorLaunchOutOfResources();
        break;
    case hipErrorLaunchTimeOut:
        throw detail::ErrorLaunchTimeOut();
        break;
    case hipErrorPeerAccessAlreadyEnabled:
        throw detail::ErrorPeerAccessAlreadyEnabled();
        break;
    case hipErrorPeerAccessNotEnabled:
        throw detail::ErrorPeerAccessNotEnabled();
        break;
    case hipErrorSetOnActiveProcess:
        throw detail::ErrorSetOnActiveProcess();
        break;
    case hipErrorContextIsDestroyed:
        throw detail::ErrorContextIsDestroyed();
        break;
    case hipErrorAssert:
        throw detail::ErrorAssert();
        break;
    case hipErrorHostMemoryAlreadyRegistered:
        throw detail::ErrorHostMemoryAlreadyRegistered();
        break;
    case hipErrorHostMemoryNotRegistered:
        throw detail::ErrorHostMemoryNotRegistered();
        break;
    case hipErrorLaunchFailure:
        throw detail::ErrorLaunchFailure();
        break;
    case hipErrorCooperativeLaunchTooLarge:
        throw detail::ErrorCooperativeLaunchTooLarge();
        break;
    case hipErrorNotSupported:
        throw detail::ErrorNotSupported();
        break;
    case hipErrorStreamCaptureUnsupported:
        throw detail::ErrorStreamCaptureUnsupported();
        break;
    case hipErrorStreamCaptureInvalidated:
        throw detail::ErrorStreamCaptureInvalidated();
        break;
    case hipErrorStreamCaptureMerge:
        throw detail::ErrorStreamCaptureMerge();
        break;
    case hipErrorStreamCaptureUnmatched:
        throw detail::ErrorStreamCaptureUnmatched();
        break;
    case hipErrorStreamCaptureUnjoined:
        throw detail::ErrorStreamCaptureUnjoined();
        break;
    case hipErrorStreamCaptureIsolation:
        throw detail::ErrorStreamCaptureIsolation();
        break;
    case hipErrorStreamCaptureImplicit:
        throw detail::ErrorStreamCaptureImplicit();
        break;
    case hipErrorCapturedEvent:
        throw detail::ErrorCapturedEvent();
        break;
    case hipErrorStreamCaptureWrongThread:
        throw detail::ErrorStreamCaptureWrongThread();
        break;
    case hipErrorGraphExecUpdateFailure:
        throw detail::ErrorGraphExecUpdateFailure();
        break;
    case hipErrorUnknown:
        throw detail::ErrorUnknown();
        break;
    case hipErrorRuntimeMemory:
        throw detail::ErrorRuntimeMemory();
        break;
    case hipErrorRuntimeOther:
        throw detail::ErrorRuntimeOther();
        break;
    case hipErrorTbd:
        throw detail::ErrorTbd();
        break;
    default :
        throw detail::ErrorUnknown();
    }
}

void initialize() {
    hip_catch(hipFree(nullptr));

    std::atexit(finalize);
}

void finalize() {
    hip_catch(hipDeviceReset());
}

END_EINSUMS_NAMESPACE_CPP(einsums::gpu)