..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.gpu_except :

************************
HIP Exceptions in Python
************************

.. sectionauthor:: Connor Briggs

.. py:currentmodule:: einsums.gpu_except

This module contains all possible exceptions that can be thrown by HIP and related libraries,
along with a brief description of each.

.. py:exception:: Success

    This can be thrown when a HIP function exits successfully. It is normally suppressed, but can
    be thrown if requested.

.. py:exception:: ErrorInvalidValue

    Indicates that at least one of the arguments passed to a function is :code:`NULL`.

.. py:exception:: ErrorOutOfMemory

    Indicates that the requested allocation could not be completed due to not having enough
    free memory to accomodate.

.. py:exception:: ErrorMemoryAllocation
    :deprecated:

    Indicates a memory allocation error. It is superceded by :py:exc:`ErrorOutOfMemory`.

.. py:exception:: ErrorNotInitialized

    This is thrown when the GPU device has not been initialized, such as by a call to :code:`hipFree(nullptr);`.

.. py:exception:: ErrorInitializationError
    :deprecated:

    Superceded by :py:exc:`ErrorNotInitialized`.

.. py:exception:: ErrorDeinitialized

    This is thrown if an operation is attempted after a device has been de-initialized, such as by a call to
    :code:`hipDeviceReset`.

.. py:exception:: ErrorProfilerDisabled

    This is thrown if an attempt is made to profile an application, but the profiler has been disabled.

.. py:exception:: ErrorProfilerNotInitialized

    This is thrown if an attempt is made to profile an application, but the profiler has not yet been initialized.

.. py:exception:: ErrorProfilerAlreadyStarted

    This is thrown if the code tries to start the profiler, but the profiler is already running.

.. py:exception:: ErrorProfilerAlreadyStopped

    This is thrown if the code tries to stop the profiler, but it has already been stopped.

.. py:exception:: ErrorInvalidConfiguration

    This is thrown when the device is not configured for a certain operation, such as interprocess communication.

.. py:exception:: ErrorInvalidPitchValue

    Indicates that the pitch of an array is invalid for some reason.

.. py:exception:: ErrorInvalidSymbol

    Thrown when an operation tries to access a symbol that is not valid for some reason, such as being null,
    or not being allocated.

.. py:exception:: ErrorInvalidDevicePointer

    Thrown when a function expects a device pointer, but the pointer provided is invalid. A common issue that
    can cause this is passing a host pointer instead of a device pointer.

.. py:exception:: ErrorInvalidMemcpyDirection

    Thrown when a :code:`hipMemcpy` is initiated, but the pointers are not of the right occupancy for the
    direction specified.

.. py:exception:: ErrorInsufficientDriver

    This is thrown when an operation is attempted that is not supported by the current device driver.

.. py:exception:: ErrorMissingConfiguration

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorPriorLaunchFailure

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorInvalidDeviceFunction

    Thrown when an attempt is made to access or modify a device function, but the requested function is invalid
    for some reason.

.. py:exception:: ErrorNoDevice

    This is thrown when no device can be found to run code.

.. py:exception:: ErrorInvalidDevice

    This is thrown when trying to access a device with an ID outside of the range of enumerated devices.

.. py:exception:: ErrorInvalidImage

    Thrown when a cooperative group is launched with an invalid image. This is likely due to the code being
    compiled for the wrong architecture.

.. py:exception:: ErrorInvalidContext

    "Produced when input context is invalid" (from :code:`hip_runtime_api.h`). This is often due to using a
    handle to something that is not defined in the current context.

.. py:exception:: ErrorContextAlreadyCurrent

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorMapFailed

    Thrown when an attempt to map some portion of memory into some virtual address space fails.

.. py:exception:: ErrorMapBufferObjectFailed
    :deprecated:

    Superceded by :py:exc:`ErrorMapFailed`.

.. py:exception:: ErrorUnmapFailed

    Thrown when an attempt to unmap some portion of virtual memory fails. For instance, if it has already been unmapped,
    this may be thrown.

.. py:exception:: ErrorArrayIsMapped

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorAlreadyMapped

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorNoBinaryForGpu

    Raised when there are multiple devices on the system, but a binary can not be found for some of them.

.. py:exception:: ErrorAlreadyAcquired

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorNotMapped

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorNotMappedAsArray

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorNotMappedAsPointer

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorECCNotCorrectable

    Unused as of now. Likely thrown when a response on the PCI bus fails the error correction check in a way
    that is detectable but not correctable.

.. py:exception:: ErrorUnsupportedLimit

    Thrown when trying to get or set a limit that is not supported by the device.

.. py:exception:: ErrorContextAlreadyInUse

    Thrown when trying to modify a context currently in use. Deprecated on AMD only.

.. py:exception:: ErrorPeerAccessUnsupported

    Thrown when trying to access a device on a multi-device system, but the operation is unsupported by either devices or the driver.

.. py:exception:: ErrorInvalidKernelFile

    Unkown as of now. This seems to be unused but not deprecated. There is also a note in
    :code:`hip_runtime_api.h - In CUDA DRV, it is CUDA_ERROR_PTX`

.. py:exception:: ErrorInvalidGraphicsContext

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorInvalidSource

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorFileNotFound

    Thrown when trying to load data from a module, but the file can not be found.

.. py:exception:: ErrorSharedObjectSymbolNotFound

    Thrown when trying to load data from a module, but the requested symbol is not found in the module.

.. py:exception:: ErrorSharedObjectInitFailed

    Thrown when trying to initialize a shared object library, but an error occurs.

.. py:exception:: ErrorOperatingSystem

    Thrown when a binary is built for one operating system but being run on a different system.

.. py:exception:: ErrorInvalidHandle

    Thrown when using a handle for an event or stream that is invalid for some reason.

.. py:exception:: ErrorInvalidResourceHandle
    :deprecated:
    
    Superceded by ErrorInvalidHandle

.. py:exception:: ErrorIllegalState

    Thrown when a resource is in a state that does not support the requested operation.

.. py:exception:: ErrorNotFound

    Thrown when trying to find a variable in a module, but the variable can not be found.

.. py:exception:: ErrorNotReady

    Thrown when trying to query an event's properties when those properties haven't been computed yet.
    For instance, trying to find the elapsed time, but the event is still running. This is not actually
    an error, according to the HIP documentation.

.. py:exception:: ErrorIllegalAddress

    Unknown as of now. This seems to be unused but not deprecated.

.. py:exception:: ErrorLaunchOutOfResources

    This is thrown when trying to launch a cooperative kernel, but there are not enough resources available for
    the kernel to run.

.. py:exception:: ErrorLaunchTimeOut

    This is thrown when trying to launch a cooperative kernel, but one of the devices times out.

.. py:exception:: ErrorPeerAccessAlreadyEnabled

    Thrown when trying to enable peer access on the current device, but it has already been enabled.

.. py:exception:: ErrorPeerAccessNotEnabled

    Thrown when trying to access a peer device, but peer access has not been enabled.

.. py:exception:: ErrorSetOnActiveProcess

    Not well documented. Thrown when trying to set flags for how the device should behave while the device is
    running a process that would be affected by those changes.

.. py:exception:: ErrorContextIsDestroyed

    Thrown when trying to destroy a context, but the context has already been destroyed.

.. py:exception:: ErrorAssert

    Thrown when an internal assertion fails.

.. py:exception:: ErrorHostMemoyrAlreadyRegistered

    Thrown when trying to register host memory, but that memory has already been registered.

.. py:exception:: ErrorHostMemoryNotRegistered
    
    Thrown when trying to unregister host memory, but that memory has not been registered or is already unregistered.

.. py:exception:: ErrorLaunchFailure

    Thrown when something goes wrong when launching a kernel.

.. py:exception:: ErrorCooperativeLaunchTooLarge

    Thrown when trying to launch a cooperative kernel with too many blocks.

.. py:exception:: ErrorNotSupported

    Thrown when the HIP API is not supported.

.. py:exception:: ErrorStreamCaptureUnsupported

    Thrown when trying to perform an operation that is not allowed when a stream is capturing.

.. py:exception:: ErrorStreamCaptureInvalidated

    Thrown when a previous error has invalidated a stream capture.

.. py:exception:: ErrorStreamCaptureMerge

    Thrown when an operation would have merged two independent stream captures.

.. py:exception:: ErrorStreamCaptureUnmatched

    Thrown when a stream tries to use a capture that was created for another stream.

.. py:exception:: ErrorStreamCaptureUnjoined

    Indicates that a stream capture was forked and never rejoined.

.. py:exception:: ErrorStreamCaptureIsolation

    Thrown when a cross-stream dependency would have been created.

.. py:exception:: ErrorStreamCaptureImplicit

    Thrown when an operation would have caused a disallowed implicit dependency in a capture.

.. py:exception:: ErrorCapturedEvent

    Indicates that an operation was not permitted on an event in a stream that is actively capturing.

.. py:exception:: ErrorStreamCaptureWrongThread

    A capture operation was initiated on a thread that does not have access to the stream.


.. py:exception:: ErrorGraphExecUpdateFailure

    This is thrown when a graph update would have violated certain constraints.

.. py:exception:: ErrorUnknown

    Thrown when an unknown error has occurred.

.. py:exception:: ErrorRuntimeMemory

    An internal memory call produced an error. Not seen on production systems.

.. py:exception:: ErrorRuntimeOther

    An internal call that is not a memory call produced an error. Not seen on production systems.

.. py:exception:: ErrorTbd

    Placeholder error for future expansion.

.. py:exception:: blasSuccess

    This can be thrown when a hipBLAS function exits successfully. It is normally suppressed, but can
    be thrown if requested.

.. py:exception:: blasNotInitialized

    Thrown when trying to perform a BLAS operation when the environment has not been initialized.

.. py:exception:: blasAllocFailed

    Thrown when an operation tries to allocate memory, such as a work array, but the allocation returned an
    invalid value.

.. py:exception:: blasInvalidValue

    Thrown when invalid values are passed to a BLAS call. For instance, many operations have constraints on
    strides or increment values. This would be thrown if those constraints are violated.

.. py:exception:: blasMappingError

    Thrown when an attempt to map memory into some virtual address space fails.

.. py:exception:: blasExecutionFailed

    Thrown when a BLAS operation fails for some reason.

.. py:exception:: blasInternalError

    Thrown when an internal operation fails.

.. py:exception:: blasNotSupported

    Thrown when an operation is not supported. For instance, not all systems support half-precision BLAS calls.

.. py:exception:: blasArchMismatch

    Thrown when hipBLAS was compiled for one architecture but is being used for another.

.. py:exception:: blasHandleIsNullptr

    Thrown when :code:`nullptr` is passed as the handle for a BLAS operation.

.. py:exception:: blasInvalidEnum

    Thrown when an invalid value is passed to an enum argument. For instance, when calling :code:`gemm`,
    BLAS expects to be told either to not transpose a matrix, transpose a matrix, or conjugate and transpose
    a matrix. Any other value would cause this error to be thrown.

.. py:exception:: blasUnknown

    Thrown when an unknown error occurs.

.. py:exception:: solverSuccess

    This can be thrown when a hipSolver function exits successfully. It is normally suppressed, but can
    be thrown if requested.

.. py:exception:: solverNotInitialized

    Thrown when trying to perform a LAPACK operation when the environment has not been initialized.

.. py:exception:: solverAllocFailed

    Thrown when an operation tries to allocate memory, such as a work array, but the allocation returned an
    invalid value.

.. py:exception:: solverInvalidValue

    Thrown when invalid values are passed to a LAPACK call. For instance, many operations have constraints on
    strides or increment values. This would be thrown if those constraints are violated.

.. py:exception:: solverMappingError

    Thrown when an attempt to map memory into some virtual address space fails.

.. py:exception:: solverExecutionFailed

    Thrown when a LAPACK operation fails for some reason.

.. py:exception:: solverInternalError

    Thrown when an internal operation fails.

.. py:exception:: solverFuncNotSupported

    Thrown when an operation is not supported. For instance, not all systems support half-precision LAPACK calls.

.. py:exception:: solverArchMismatch

    Thrown when hipSolver was compiled for one architecture but is being used for another.

.. py:exception:: solverHandleIsNullptr

    Thrown when :code:`nullptr` is passed as the handle for a LAPACK operation.

.. py:exception:: solverInvalidEnum

    Thrown when an invalid value is passed to an enum argument. For instance, when calling :code:`geev`,
    LAPACK expects to be told whether or not to compute the left or right eigenvectors.
    Any other value would cause this error to be thrown.

.. py:exception:: solverUnknown

    Thrown when an unknown error occurs.