#include <Einsums/Errors/Error.hpp>

namespace einsums {
EINSUMS_HOST void __hip_catch__(hipError_t condition, char const *func_call, char const *fname, char const *location, char const *funcname,
                                bool throw_success) {

    std::string diagnostic = fname;
    diagnostic += location;
    diagnostic.append(funcname);
    diagnostic += "\nError in line: ";
    diagnostic += func_call;
    diagnostic += "\nHIP error code ";
    diagnostic += std::to_string((int)condition);
    diagnostic += ": ";

    // Auto-generated code.
    switch (condition) {
    case hipSuccess:
        if (throw_success) {
            throw Success(diagnostic);
        } else {
            return;
        }
        break;
    case hipErrorInvalidValue:
        throw ErrorInvalidValue(diagnostic);
        break;
    case hipErrorOutOfMemory:
        throw ErrorOutOfMemory(diagnostic);
        break;
    case hipErrorNotInitialized:
        throw ErrorNotInitialized(diagnostic);
        break;
    case hipErrorDeinitialized:
        throw ErrorDeinitialized(diagnostic);
        break;
    case hipErrorProfilerDisabled:
        throw ErrorProfilerDisabled(diagnostic);
        break;
    case hipErrorProfilerNotInitialized:
        throw ErrorProfilerNotInitialized(diagnostic);
        break;
    case hipErrorProfilerAlreadyStarted:
        throw ErrorProfilerAlreadyStarted(diagnostic);
        break;
    case hipErrorProfilerAlreadyStopped:
        throw ErrorProfilerAlreadyStopped(diagnostic);
        break;
    case hipErrorInvalidConfiguration:
        throw ErrorInvalidConfiguration(diagnostic);
        break;
    case hipErrorInvalidPitchValue:
        throw ErrorInvalidPitchValue(diagnostic);
        break;
    case hipErrorInvalidSymbol:
        throw ErrorInvalidSymbol(diagnostic);
        break;
    case hipErrorInvalidDevicePointer:
        throw ErrorInvalidDevicePointer(diagnostic);
        break;
    case hipErrorInvalidMemcpyDirection:
        throw ErrorInvalidMemcpyDirection(diagnostic);
        break;
    case hipErrorInsufficientDriver:
        throw ErrorInsufficientDriver(diagnostic);
        break;
    case hipErrorMissingConfiguration:
        throw ErrorMissingConfiguration(diagnostic);
        break;
    case hipErrorPriorLaunchFailure:
        throw ErrorPriorLaunchFailure(diagnostic);
        break;
    case hipErrorInvalidDeviceFunction:
        throw ErrorInvalidDeviceFunction(diagnostic);
        break;
    case hipErrorNoDevice:
        throw ErrorNoDevice(diagnostic);
        break;
    case hipErrorInvalidDevice:
        throw ErrorInvalidDevice(diagnostic);
        break;
    case hipErrorInvalidImage:
        throw ErrorInvalidImage(diagnostic);
        break;
    case hipErrorInvalidContext:
        throw ErrorInvalidContext(diagnostic);
        break;
    case hipErrorContextAlreadyCurrent:
        throw ErrorContextAlreadyCurrent(diagnostic);
        break;
    case hipErrorMapFailed:
        throw ErrorMapFailed(diagnostic);
        break;
    case hipErrorUnmapFailed:
        throw ErrorUnmapFailed(diagnostic);
        break;
    case hipErrorArrayIsMapped:
        throw ErrorArrayIsMapped(diagnostic);
        break;
    case hipErrorAlreadyMapped:
        throw ErrorAlreadyMapped(diagnostic);
        break;
    case hipErrorNoBinaryForGpu:
        throw ErrorNoBinaryForGpu(diagnostic);
        break;
    case hipErrorAlreadyAcquired:
        throw ErrorAlreadyAcquired(diagnostic);
        break;
    case hipErrorNotMapped:
        throw ErrorNotMapped(diagnostic);
        break;
    case hipErrorNotMappedAsArray:
        throw ErrorNotMappedAsArray(diagnostic);
        break;
    case hipErrorNotMappedAsPointer:
        throw ErrorNotMappedAsPointer(diagnostic);
        break;
    case hipErrorECCNotCorrectable:
        throw ErrorECCNotCorrectable(diagnostic);
        break;
    case hipErrorUnsupportedLimit:
        throw ErrorUnsupportedLimit(diagnostic);
        break;
    case hipErrorContextAlreadyInUse:
        throw ErrorContextAlreadyInUse(diagnostic);
        break;
    case hipErrorPeerAccessUnsupported:
        throw ErrorPeerAccessUnsupported(diagnostic);
        break;
    case hipErrorInvalidKernelFile:
        throw ErrorInvalidKernelFile(diagnostic);
        break;
    case hipErrorInvalidGraphicsContext:
        throw ErrorInvalidGraphicsContext(diagnostic);
        break;
    case hipErrorInvalidSource:
        throw ErrorInvalidSource(diagnostic);
        break;
    case hipErrorFileNotFound:
        throw ErrorFileNotFound(diagnostic);
        break;
    case hipErrorSharedObjectSymbolNotFound:
        throw ErrorSharedObjectSymbolNotFound(diagnostic);
        break;
    case hipErrorSharedObjectInitFailed:
        throw ErrorSharedObjectInitFailed(diagnostic);
        break;
    case hipErrorOperatingSystem:
        throw ErrorOperatingSystem(diagnostic);
        break;
    case hipErrorInvalidHandle:
        throw ErrorInvalidHandle(diagnostic);
        break;
    case hipErrorIllegalState:
        throw ErrorIllegalState(diagnostic);
        break;
    case hipErrorNotFound:
        throw ErrorNotFound(diagnostic);
        break;
    case hipErrorNotReady:
        throw ErrorNotReady(diagnostic);
        break;
    case hipErrorIllegalAddress:
        throw ErrorIllegalAddress(diagnostic);
        break;
    case hipErrorLaunchOutOfResources:
        throw ErrorLaunchOutOfResources(diagnostic);
        break;
    case hipErrorLaunchTimeOut:
        throw ErrorLaunchTimeOut(diagnostic);
        break;
    case hipErrorPeerAccessAlreadyEnabled:
        throw ErrorPeerAccessAlreadyEnabled(diagnostic);
        break;
    case hipErrorPeerAccessNotEnabled:
        throw ErrorPeerAccessNotEnabled(diagnostic);
        break;
    case hipErrorSetOnActiveProcess:
        throw ErrorSetOnActiveProcess(diagnostic);
        break;
    case hipErrorContextIsDestroyed:
        throw ErrorContextIsDestroyed(diagnostic);
        break;
    case hipErrorAssert:
        throw ErrorAssert(diagnostic);
        break;
    case hipErrorHostMemoryAlreadyRegistered:
        throw ErrorHostMemoryAlreadyRegistered(diagnostic);
        break;
    case hipErrorHostMemoryNotRegistered:
        throw ErrorHostMemoryNotRegistered(diagnostic);
        break;
    case hipErrorLaunchFailure:
        throw ErrorLaunchFailure(diagnostic);
        break;
    case hipErrorCooperativeLaunchTooLarge:
        throw ErrorCooperativeLaunchTooLarge(diagnostic);
        break;
    case hipErrorNotSupported:
        throw ErrorNotSupported(diagnostic);
        break;
    case hipErrorStreamCaptureUnsupported:
        throw ErrorStreamCaptureUnsupported(diagnostic);
        break;
    case hipErrorStreamCaptureInvalidated:
        throw ErrorStreamCaptureInvalidated(diagnostic);
        break;
    case hipErrorStreamCaptureMerge:
        throw ErrorStreamCaptureMerge(diagnostic);
        break;
    case hipErrorStreamCaptureUnmatched:
        throw ErrorStreamCaptureUnmatched(diagnostic);
        break;
    case hipErrorStreamCaptureUnjoined:
        throw ErrorStreamCaptureUnjoined(diagnostic);
        break;
    case hipErrorStreamCaptureIsolation:
        throw ErrorStreamCaptureIsolation(diagnostic);
        break;
    case hipErrorStreamCaptureImplicit:
        throw ErrorStreamCaptureImplicit(diagnostic);
        break;
    case hipErrorCapturedEvent:
        throw ErrorCapturedEvent(diagnostic);
        break;
    case hipErrorStreamCaptureWrongThread:
        throw ErrorStreamCaptureWrongThread(diagnostic);
        break;
    case hipErrorGraphExecUpdateFailure:
        throw ErrorGraphExecUpdateFailure(diagnostic);
        break;
    case hipErrorUnknown:
        throw ErrorUnknown(diagnostic);
        break;
    case hipErrorRuntimeMemory:
        throw ErrorRuntimeMemory(diagnostic);
        break;
    case hipErrorRuntimeOther:
        throw ErrorRuntimeOther(diagnostic);
        break;
    case hipErrorTbd:
        throw ErrorTbd(diagnostic);
        break;
    default:
        throw ErrorUnknown(diagnostic);
    }
}

EINSUMS_HOST void __hipblas_catch__(hipblasStatus_t status, char const *func_call, char const *fname, char const *location,
                                    char const *funcname, bool throw_success) {
    std::string diagnostic = fname;
    diagnostic += location;
    diagnostic += "\nError in line: ";
    diagnostic += func_call;
    diagnostic += "\nhipBLAS error code ";
    diagnostic += std::to_string((int)status);
    diagnostic += ": ";

    diagnostic.append(funcname);

    switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
        if (throw_success) {
            throw blasSuccess(diagnostic);
        } else {
            return;
        }
        break;
    case HIPBLAS_STATUS_NOT_INITIALIZED:
        throw blasNotInitialized(diagnostic);
        break;
    case HIPBLAS_STATUS_ALLOC_FAILED:
        throw blasAllocFailed(diagnostic);
        break;
    case HIPBLAS_STATUS_INVALID_VALUE:
        throw blasInvalidValue(diagnostic);
        break;
    case HIPBLAS_STATUS_MAPPING_ERROR:
        throw blasMappingError(diagnostic);
        break;
    case HIPBLAS_STATUS_EXECUTION_FAILED:
        throw blasExecutionFailed(diagnostic);
        break;
    case HIPBLAS_STATUS_INTERNAL_ERROR:
        throw blasInternalError(diagnostic);
        break;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
        throw blasNotSupported(diagnostic);
        break;
    case HIPBLAS_STATUS_ARCH_MISMATCH:
        throw blasArchMismatch(diagnostic);
        break;
    case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
        throw blasHandleIsNullptr(diagnostic);
        break;
    case HIPBLAS_STATUS_INVALID_ENUM:
        throw blasInvalidEnum(diagnostic);
        break;
    default:
        throw blasUnknown(diagnostic);
    }
}

EINSUMS_HOST void __hipsolver_catch__(hipsolverStatus_t condition, char const *func_call, char const *fname, char const *location,
                                      char const *funcname, bool throw_success) {
    std::string diagnostic = fname;
    diagnostic += location;
    diagnostic.append(funcname);
    diagnostic += "\nError in line: ";
    diagnostic += func_call;
    diagnostic += "\nhipSolver error code ";
    diagnostic += std::to_string((int)condition);
    diagnostic += ": ";

    switch (condition) {
    case HIPSOLVER_STATUS_SUCCESS:
        if (throw_success) {
            throw solverSuccess(diagnostic);
        } else {
            return;
        }
        break;
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
        throw solverNotInitialized(diagnostic);
        break;
    case HIPSOLVER_STATUS_ALLOC_FAILED:
        throw solverAllocFailed(diagnostic);
        break;
    case HIPSOLVER_STATUS_INVALID_VALUE:
        throw solverInvalidValue(diagnostic);
        break;
    case HIPSOLVER_STATUS_MAPPING_ERROR:
        throw solverMappingError(diagnostic);
        break;
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
        throw solverExecutionFailed(diagnostic);
        break;
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
        throw solverInternalError(diagnostic);
        break;
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
        throw solverFuncNotSupported(diagnostic);
        break;
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
        throw solverArchMismatch(diagnostic);
        break;
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
        throw solverHandleIsNullptr(diagnostic);
        break;
    case HIPSOLVER_STATUS_INVALID_ENUM:
        throw solverInvalidEnum(diagnostic);
        break;
    case HIPSOLVER_STATUS_UNKNOWN:
        throw solverUnknown(diagnostic);
        break;
    // case HIPSOLVER_STATUS_ZERO_PIVOT:
    //     throw solverZeroPivot();
    //     break;
    default:
        throw solverUnknown(diagnostic);
    }
}

char const *hipsolverStatusToString(hipsolverStatus_t status) {
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

} // namespace einsums