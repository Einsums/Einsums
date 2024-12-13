#include <Einsums/Config.hpp>

#include <Einsums/Errors/Error.hpp>

#include <EinsumsPy/Errors/Export.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define EXCEPTION(name) py::register_exception<einsums::name>(error_mod, #name)

void export_Errors(py::module_ &mod) {

    py::module_ error_mod = mod.def_submodule("errors", "This module contains all possible errors that can be thrown by Einsums.");

    EXCEPTION(dimension_error);
    EXCEPTION(tensor_compat_error);
    EXCEPTION(num_argument_error);
    EXCEPTION(not_enough_args);
    EXCEPTION(too_many_args);
    EXCEPTION(access_denied);
    EXCEPTION(todo_error);
    EXCEPTION(bad_logic);
    EXCEPTION(uninitialized_error);
    EXCEPTION(enum_error);

#ifdef EINSUMS_COMPUTE_CODE
    py::register_exception<einsums::Success>(error_mod, "Success");
    py::register_exception<einsums::ErrorInvalidValue>(error_mod, "ErrorInvalidValue");
    py::register_exception<einsums::ErrorOutOfMemory>(error_mod, "ErrorOutOfMemory");
    py::register_exception<einsums::ErrorMemoryAllocation>(error_mod, "ErrorMemoryAllocation");
    py::register_exception<einsums::ErrorNotInitialized>(error_mod, "ErrorNotInitialized");
    py::register_exception<einsums::ErrorInitializationError>(error_mod, "ErrorInitializationError");
    py::register_exception<einsums::ErrorDeinitialized>(error_mod, "ErrorDeinitialized");
    py::register_exception<einsums::ErrorProfilerDisabled>(error_mod, "ErrorProfilerDisabled");
    py::register_exception<einsums::ErrorProfilerNotInitialized>(error_mod, "ErrorProfilerNotInitialized");
    py::register_exception<einsums::ErrorProfilerAlreadyStarted>(error_mod, "ErrorProfilerAlreadyStarted");
    py::register_exception<einsums::ErrorProfilerAlreadyStopped>(error_mod, "ErrorProfilerAlreadyStopped");
    py::register_exception<einsums::ErrorInvalidConfiguration>(error_mod, "ErrorInvalidConfiguration");
    py::register_exception<einsums::ErrorInvalidPitchValue>(error_mod, "ErrorInvalidPitchValue");
    py::register_exception<einsums::ErrorInvalidSymbol>(error_mod, "ErrorInvalidSymbol");
    py::register_exception<einsums::ErrorInvalidDevicePointer>(error_mod, "ErrorInvalidDevicePointer");
    py::register_exception<einsums::ErrorInvalidMemcpyDirection>(error_mod, "ErrorInvalidMemcpyDirection");
    py::register_exception<einsums::ErrorInsufficientDriver>(error_mod, "ErrorInsufficientDriver");
    py::register_exception<einsums::ErrorMissingConfiguration>(error_mod, "ErrorMissingConfiguration");
    py::register_exception<einsums::ErrorPriorLaunchFailure>(error_mod, "ErrorPriorLaunchFailure");
    py::register_exception<einsums::ErrorInvalidDeviceFunction>(error_mod, "ErrorInvalidDeviceFunction");
    py::register_exception<einsums::ErrorNoDevice>(error_mod, "ErrorNoDevice");
    py::register_exception<einsums::ErrorInvalidDevice>(error_mod, "ErrorInvalidDevice");
    py::register_exception<einsums::ErrorInvalidImage>(error_mod, "ErrorInvalidImage");
    py::register_exception<einsums::ErrorInvalidContext>(error_mod, "ErrorInvalidContext");
    py::register_exception<einsums::ErrorContextAlreadyCurrent>(error_mod, "ErrorContextAlreadyCurrent");
    py::register_exception<einsums::ErrorMapFailed>(error_mod, "ErrorMapFailed");
    py::register_exception<einsums::ErrorMapBufferObjectFailed>(error_mod, "ErrorMapBufferObjectFailed");
    py::register_exception<einsums::ErrorUnmapFailed>(error_mod, "ErrorUnmapFailed");
    py::register_exception<einsums::ErrorArrayIsMapped>(error_mod, "ErrorArrayIsMapped");
    py::register_exception<einsums::ErrorAlreadyMapped>(error_mod, "ErrorAlreadyMapped");
    py::register_exception<einsums::ErrorNoBinaryForGpu>(error_mod, "ErrorNoBinaryForGpu");
    py::register_exception<einsums::ErrorAlreadyAcquired>(error_mod, "ErrorAlreadyAcquired");
    py::register_exception<einsums::ErrorNotMapped>(error_mod, "ErrorNotMapped");
    py::register_exception<einsums::ErrorNotMappedAsArray>(error_mod, "ErrorNotMappedAsArray");
    py::register_exception<einsums::ErrorNotMappedAsPointer>(error_mod, "ErrorNotMappedAsPointer");
    py::register_exception<einsums::ErrorECCNotCorrectable>(error_mod, "ErrorECCNotCorrectable");
    py::register_exception<einsums::ErrorUnsupportedLimit>(error_mod, "ErrorUnsupportedLimit");
    py::register_exception<einsums::ErrorContextAlreadyInUse>(error_mod, "ErrorContextAlreadyInUse");
    py::register_exception<einsums::ErrorPeerAccessUnsupported>(error_mod, "ErrorPeerAccessUnsupported");
    py::register_exception<einsums::ErrorInvalidKernelFile>(error_mod, "ErrorInvalidKernelFile");
    py::register_exception<einsums::ErrorInvalidGraphicsContext>(error_mod, "ErrorInvalidGraphicsContext");
    py::register_exception<einsums::ErrorInvalidSource>(error_mod, "ErrorInvalidSource");
    py::register_exception<einsums::ErrorFileNotFound>(error_mod, "ErrorFileNotFound");
    py::register_exception<einsums::ErrorSharedObjectSymbolNotFound>(error_mod, "ErrorSharedObjectSymbolNotFound");
    py::register_exception<einsums::ErrorSharedObjectInitFailed>(error_mod, "ErrorSharedObjectInitFailed");
    py::register_exception<einsums::ErrorOperatingSystem>(error_mod, "ErrorOperatingSystem");
    py::register_exception<einsums::ErrorInvalidHandle>(error_mod, "ErrorInvalidHandle");
    py::register_exception<einsums::ErrorInvalidResourceHandle>(error_mod, "ErrorInvalidResourceHandle");
    py::register_exception<einsums::ErrorIllegalState>(error_mod, "ErrorIllegalState");
    py::register_exception<einsums::ErrorNotFound>(error_mod, "ErrorNotFound");
    py::register_exception<einsums::ErrorNotReady>(error_mod, "ErrorNotReady");
    py::register_exception<einsums::ErrorIllegalAddress>(error_mod, "ErrorIllegalAddress");
    py::register_exception<einsums::ErrorLaunchOutOfResources>(error_mod, "ErrorLaunchOutOfResources");
    py::register_exception<einsums::ErrorLaunchTimeOut>(error_mod, "ErrorLaunchTimeOut");
    py::register_exception<einsums::ErrorPeerAccessAlreadyEnabled>(error_mod, "ErrorPeerAccessAlreadyEnabled");
    py::register_exception<einsums::ErrorPeerAccessNotEnabled>(error_mod, "ErrorPeerAccessNotEnabled");
    py::register_exception<einsums::ErrorSetOnActiveProcess>(error_mod, "ErrorSetOnActiveProcess");
    py::register_exception<einsums::ErrorContextIsDestroyed>(error_mod, "ErrorContextIsDestroyed");
    py::register_exception<einsums::ErrorAssert>(error_mod, "ErrorAssert");
    py::register_exception<einsums::ErrorHostMemoryAlreadyRegistered>(error_mod, "ErrorHostMemoryAlreadyRegistered");
    py::register_exception<einsums::ErrorHostMemoryNotRegistered>(error_mod, "ErrorHostMemoryNotRegistered");
    py::register_exception<einsums::ErrorLaunchFailure>(error_mod, "ErrorLaunchFailure");
    py::register_exception<einsums::ErrorCooperativeLaunchTooLarge>(error_mod, "ErrorCooperativeLaunchTooLarge");
    py::register_exception<einsums::ErrorNotSupported>(error_mod, "ErrorNotSupported");
    py::register_exception<einsums::ErrorStreamCaptureUnsupported>(error_mod, "ErrorStreamCaptureUnsupported");
    py::register_exception<einsums::ErrorStreamCaptureInvalidated>(error_mod, "ErrorStreamCaptureInvalidated");
    py::register_exception<einsums::ErrorStreamCaptureMerge>(error_mod, "ErrorStreamCaptureMerge");
    py::register_exception<einsums::ErrorStreamCaptureUnmatched>(error_mod, "ErrorStreamCaptureUnmatched");
    py::register_exception<einsums::ErrorStreamCaptureUnjoined>(error_mod, "ErrorStreamCaptureUnjoined");
    py::register_exception<einsums::ErrorStreamCaptureIsolation>(error_mod, "ErrorStreamCaptureIsolation");
    py::register_exception<einsums::ErrorStreamCaptureImplicit>(error_mod, "ErrorStreamCaptureImplicit");
    py::register_exception<einsums::ErrorCapturedEvent>(error_mod, "ErrorCapturedEvent");
    py::register_exception<einsums::ErrorStreamCaptureWrongThread>(error_mod, "ErrorStreamCaptureWrongThread");
    py::register_exception<einsums::ErrorGraphExecUpdateFailure>(error_mod, "ErrorGraphExecUpdateFailure");
    py::register_exception<einsums::ErrorUnknown>(error_mod, "ErrorUnknown");
    py::register_exception<einsums::ErrorRuntimeMemory>(error_mod, "ErrorRuntimeMemory");
    py::register_exception<einsums::ErrorRuntimeOther>(error_mod, "ErrorRuntimeOther");
    py::register_exception<einsums::ErrorTbd>(error_mod, "ErrorTbd");
    py::register_exception<einsums::blasSuccess>(error_mod, "blasSuccess");
    py::register_exception<einsums::blasNotInitialized>(error_mod, "blasNotInitialized");
    py::register_exception<einsums::blasAllocFailed>(error_mod, "blasAllocFailed");
    py::register_exception<einsums::blasInvalidValue>(error_mod, "blasInvalidValue");
    py::register_exception<einsums::blasMappingError>(error_mod, "blasMappingError");
    py::register_exception<einsums::blasExecutionFailed>(error_mod, "blasExecutionFailed");
    py::register_exception<einsums::blasInternalError>(error_mod, "blasInternalError");
    py::register_exception<einsums::blasNotSupported>(error_mod, "blasNotSupported");
    py::register_exception<einsums::blasArchMismatch>(error_mod, "blasArchMismatch");
    py::register_exception<einsums::blasHandleIsNullptr>(error_mod, "blasHandleIsNullptr");
    py::register_exception<einsums::blasInvalidEnum>(error_mod, "blasInvalidEnum");
    py::register_exception<einsums::blasUnknown>(error_mod, "blasUnknown");
    py::register_exception<einsums::solverSuccess>(error_mod, "solverSuccess");
    py::register_exception<einsums::solverNotInitialized>(error_mod, "solverNotInitialized");
    py::register_exception<einsums::solverAllocFailed>(error_mod, "solverAllocFailed");
    py::register_exception<einsums::solverInvalidValue>(error_mod, "solverInvalidValue");
    py::register_exception<einsums::solverMappingError>(error_mod, "solverMappingError");
    py::register_exception<einsums::solverExecutionFailed>(error_mod, "solverExecutionFailed");
    py::register_exception<einsums::solverInternalError>(error_mod, "solverInternalError");
    py::register_exception<einsums::solverFuncNotSupported>(error_mod, "solverFuncNotSupported");
    py::register_exception<einsums::solverArchMismatch>(error_mod, "solverArchMismatch");
    py::register_exception<einsums::solverHandleIsNullptr>(error_mod, "solverHandleIsNullptr");
    py::register_exception<einsums::solverInvalidEnum>(error_mod, "solverInvalidEnum");
    py::register_exception<einsums::solverUnknown>(error_mod, "solverUnknown");

    py::register_local_exception<einsums::Success>(error_mod, "Success");
    py::register_local_exception<einsums::ErrorInvalidValue>(error_mod, "ErrorInvalidValue");
    py::register_local_exception<einsums::ErrorOutOfMemory>(error_mod, "ErrorOutOfMemory");
    py::register_local_exception<einsums::ErrorMemoryAllocation>(error_mod, "ErrorMemoryAllocation");
    py::register_local_exception<einsums::ErrorNotInitialized>(error_mod, "ErrorNotInitialized");
    py::register_local_exception<einsums::ErrorInitializationError>(error_mod, "ErrorInitializationError");
    py::register_local_exception<einsums::ErrorDeinitialized>(error_mod, "ErrorDeinitialized");
    py::register_local_exception<einsums::ErrorProfilerDisabled>(error_mod, "ErrorProfilerDisabled");
    py::register_local_exception<einsums::ErrorProfilerNotInitialized>(error_mod, "ErrorProfilerNotInitialized");
    py::register_local_exception<einsums::ErrorProfilerAlreadyStarted>(error_mod, "ErrorProfilerAlreadyStarted");
    py::register_local_exception<einsums::ErrorProfilerAlreadyStopped>(error_mod, "ErrorProfilerAlreadyStopped");
    py::register_local_exception<einsums::ErrorInvalidConfiguration>(error_mod, "ErrorInvalidConfiguration");
    py::register_local_exception<einsums::ErrorInvalidPitchValue>(error_mod, "ErrorInvalidPitchValue");
    py::register_local_exception<einsums::ErrorInvalidSymbol>(error_mod, "ErrorInvalidSymbol");
    py::register_local_exception<einsums::ErrorInvalidDevicePointer>(error_mod, "ErrorInvalidDevicePointer");
    py::register_local_exception<einsums::ErrorInvalidMemcpyDirection>(error_mod, "ErrorInvalidMemcpyDirection");
    py::register_local_exception<einsums::ErrorInsufficientDriver>(error_mod, "ErrorInsufficientDriver");
    py::register_local_exception<einsums::ErrorMissingConfiguration>(error_mod, "ErrorMissingConfiguration");
    py::register_local_exception<einsums::ErrorPriorLaunchFailure>(error_mod, "ErrorPriorLaunchFailure");
    py::register_local_exception<einsums::ErrorInvalidDeviceFunction>(error_mod, "ErrorInvalidDeviceFunction");
    py::register_local_exception<einsums::ErrorNoDevice>(error_mod, "ErrorNoDevice");
    py::register_local_exception<einsums::ErrorInvalidDevice>(error_mod, "ErrorInvalidDevice");
    py::register_local_exception<einsums::ErrorInvalidImage>(error_mod, "ErrorInvalidImage");
    py::register_local_exception<einsums::ErrorInvalidContext>(error_mod, "ErrorInvalidContext");
    py::register_local_exception<einsums::ErrorContextAlreadyCurrent>(error_mod, "ErrorContextAlreadyCurrent");
    py::register_local_exception<einsums::ErrorMapFailed>(error_mod, "ErrorMapFailed");
    py::register_local_exception<einsums::ErrorMapBufferObjectFailed>(error_mod, "ErrorMapBufferObjectFailed");
    py::register_local_exception<einsums::ErrorUnmapFailed>(error_mod, "ErrorUnmapFailed");
    py::register_local_exception<einsums::ErrorArrayIsMapped>(error_mod, "ErrorArrayIsMapped");
    py::register_local_exception<einsums::ErrorAlreadyMapped>(error_mod, "ErrorAlreadyMapped");
    py::register_local_exception<einsums::ErrorNoBinaryForGpu>(error_mod, "ErrorNoBinaryForGpu");
    py::register_local_exception<einsums::ErrorAlreadyAcquired>(error_mod, "ErrorAlreadyAcquired");
    py::register_local_exception<einsums::ErrorNotMapped>(error_mod, "ErrorNotMapped");
    py::register_local_exception<einsums::ErrorNotMappedAsArray>(error_mod, "ErrorNotMappedAsArray");
    py::register_local_exception<einsums::ErrorNotMappedAsPointer>(error_mod, "ErrorNotMappedAsPointer");
    py::register_local_exception<einsums::ErrorECCNotCorrectable>(error_mod, "ErrorECCNotCorrectable");
    py::register_local_exception<einsums::ErrorUnsupportedLimit>(error_mod, "ErrorUnsupportedLimit");
    py::register_local_exception<einsums::ErrorContextAlreadyInUse>(error_mod, "ErrorContextAlreadyInUse");
    py::register_local_exception<einsums::ErrorPeerAccessUnsupported>(error_mod, "ErrorPeerAccessUnsupported");
    py::register_local_exception<einsums::ErrorInvalidKernelFile>(error_mod, "ErrorInvalidKernelFile");
    py::register_local_exception<einsums::ErrorInvalidGraphicsContext>(error_mod, "ErrorInvalidGraphicsContext");
    py::register_local_exception<einsums::ErrorInvalidSource>(error_mod, "ErrorInvalidSource");
    py::register_local_exception<einsums::ErrorFileNotFound>(error_mod, "ErrorFileNotFound");
    py::register_local_exception<einsums::ErrorSharedObjectSymbolNotFound>(error_mod, "ErrorSharedObjectSymbolNotFound");
    py::register_local_exception<einsums::ErrorSharedObjectInitFailed>(error_mod, "ErrorSharedObjectInitFailed");
    py::register_local_exception<einsums::ErrorOperatingSystem>(error_mod, "ErrorOperatingSystem");
    py::register_local_exception<einsums::ErrorInvalidHandle>(error_mod, "ErrorInvalidHandle");
    py::register_local_exception<einsums::ErrorInvalidResourceHandle>(error_mod, "ErrorInvalidResourceHandle");
    py::register_local_exception<einsums::ErrorIllegalState>(error_mod, "ErrorIllegalState");
    py::register_local_exception<einsums::ErrorNotFound>(error_mod, "ErrorNotFound");
    py::register_local_exception<einsums::ErrorNotReady>(error_mod, "ErrorNotReady");
    py::register_local_exception<einsums::ErrorIllegalAddress>(error_mod, "ErrorIllegalAddress");
    py::register_local_exception<einsums::ErrorLaunchOutOfResources>(error_mod, "ErrorLaunchOutOfResources");
    py::register_local_exception<einsums::ErrorLaunchTimeOut>(error_mod, "ErrorLaunchTimeOut");
    py::register_local_exception<einsums::ErrorPeerAccessAlreadyEnabled>(error_mod, "ErrorPeerAccessAlreadyEnabled");
    py::register_local_exception<einsums::ErrorPeerAccessNotEnabled>(error_mod, "ErrorPeerAccessNotEnabled");
    py::register_local_exception<einsums::ErrorSetOnActiveProcess>(error_mod, "ErrorSetOnActiveProcess");
    py::register_local_exception<einsums::ErrorContextIsDestroyed>(error_mod, "ErrorContextIsDestroyed");
    py::register_local_exception<einsums::ErrorAssert>(error_mod, "ErrorAssert");
    py::register_local_exception<einsums::ErrorHostMemoryAlreadyRegistered>(error_mod, "ErrorHostMemoryAlreadyRegistered");
    py::register_local_exception<einsums::ErrorHostMemoryNotRegistered>(error_mod, "ErrorHostMemoryNotRegistered");
    py::register_local_exception<einsums::ErrorLaunchFailure>(error_mod, "ErrorLaunchFailure");
    py::register_local_exception<einsums::ErrorCooperativeLaunchTooLarge>(error_mod, "ErrorCooperativeLaunchTooLarge");
    py::register_local_exception<einsums::ErrorNotSupported>(error_mod, "ErrorNotSupported");
    py::register_local_exception<einsums::ErrorStreamCaptureUnsupported>(error_mod, "ErrorStreamCaptureUnsupported");
    py::register_local_exception<einsums::ErrorStreamCaptureInvalidated>(error_mod, "ErrorStreamCaptureInvalidated");
    py::register_local_exception<einsums::ErrorStreamCaptureMerge>(error_mod, "ErrorStreamCaptureMerge");
    py::register_local_exception<einsums::ErrorStreamCaptureUnmatched>(error_mod, "ErrorStreamCaptureUnmatched");
    py::register_local_exception<einsums::ErrorStreamCaptureUnjoined>(error_mod, "ErrorStreamCaptureUnjoined");
    py::register_local_exception<einsums::ErrorStreamCaptureIsolation>(error_mod, "ErrorStreamCaptureIsolation");
    py::register_local_exception<einsums::ErrorStreamCaptureImplicit>(error_mod, "ErrorStreamCaptureImplicit");
    py::register_local_exception<einsums::ErrorCapturedEvent>(error_mod, "ErrorCapturedEvent");
    py::register_local_exception<einsums::ErrorStreamCaptureWrongThread>(error_mod, "ErrorStreamCaptureWrongThread");
    py::register_local_exception<einsums::ErrorGraphExecUpdateFailure>(error_mod, "ErrorGraphExecUpdateFailure");
    py::register_local_exception<einsums::ErrorUnknown>(error_mod, "ErrorUnknown");
    py::register_local_exception<einsums::ErrorRuntimeMemory>(error_mod, "ErrorRuntimeMemory");
    py::register_local_exception<einsums::ErrorRuntimeOther>(error_mod, "ErrorRuntimeOther");
    py::register_local_exception<einsums::ErrorTbd>(error_mod, "ErrorTbd");
    py::register_local_exception<einsums::blasSuccess>(error_mod, "blasSuccess");
    py::register_local_exception<einsums::blasNotInitialized>(error_mod, "blasNotInitialized");
    py::register_local_exception<einsums::blasAllocFailed>(error_mod, "blasAllocFailed");
    py::register_local_exception<einsums::blasInvalidValue>(error_mod, "blasInvalidValue");
    py::register_local_exception<einsums::blasMappingError>(error_mod, "blasMappingError");
    py::register_local_exception<einsums::blasExecutionFailed>(error_mod, "blasExecutionFailed");
    py::register_local_exception<einsums::blasInternalError>(error_mod, "blasInternalError");
    py::register_local_exception<einsums::blasNotSupported>(error_mod, "blasNotSupported");
    py::register_local_exception<einsums::blasArchMismatch>(error_mod, "blasArchMismatch");
    py::register_local_exception<einsums::blasHandleIsNullptr>(error_mod, "blasHandleIsNullptr");
    py::register_local_exception<einsums::blasInvalidEnum>(error_mod, "blasInvalidEnum");
    py::register_local_exception<einsums::blasUnknown>(error_mod, "blasUnknown");
    py::register_local_exception<einsums::solverSuccess>(error_mod, "solverSuccess");
    py::register_local_exception<einsums::solverNotInitialized>(error_mod, "solverNotInitialized");
    py::register_local_exception<einsums::solverAllocFailed>(error_mod, "solverAllocFailed");
    py::register_local_exception<einsums::solverInvalidValue>(error_mod, "solverInvalidValue");
    py::register_local_exception<einsums::solverMappingError>(error_mod, "solverMappingError");
    py::register_local_exception<einsums::solverExecutionFailed>(error_mod, "solverExecutionFailed");
    py::register_local_exception<einsums::solverInternalError>(error_mod, "solverInternalError");
    py::register_local_exception<einsums::solverFuncNotSupported>(error_mod, "solverFuncNotSupported");
    py::register_local_exception<einsums::solverArchMismatch>(error_mod, "solverArchMismatch");
    py::register_local_exception<einsums::solverHandleIsNullptr>(error_mod, "solverHandleIsNullptr");
    py::register_local_exception<einsums::solverInvalidEnum>(error_mod, "solverInvalidEnum");
    py::register_local_exception<einsums::solverUnknown>(error_mod, "solverUnknown");
#endif
}