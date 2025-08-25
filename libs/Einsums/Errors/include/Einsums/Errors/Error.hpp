//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TypeSupport/StringLiteral.hpp>

#include <source_location>
#include <stdexcept>
#include <string>

#ifdef EINSUMS_COMPUTE_CODE
#    include <hip/hip_common.h>
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#    include <hipblas/hipblas.h>
#    include <hipsolver/hipsolver.h>
#endif

namespace einsums {

namespace detail {

/**
 * Construct a message that contains the type of error being produced, the location that error is being emitted,
 * and the actual message for the error.
 *
 * @param type_name The name of the type producing the error.
 * @param str The message for the error.
 * @param location The source location that the error is being emitted.
 *
 * @return A message with this extra debugging info.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT std::string make_error_message(std::string_view const &type_name, char const *str, std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
template <size_t N>
std::string make_error_message(StringLiteral<N> const type_name, char const *str, std::source_location const &location) {
    return make_error_message(type_name.string_view(), str, location);
}

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
EINSUMS_EXPORT std::string make_error_message(std::string_view const &type_name, std::string const &str,
                                              std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
template <size_t N>
std::string make_error_message(StringLiteral<N> const type_name, std::string const &str, std::source_location const &location) {
    return make_error_message(type_name.string_view(), str, location);
}

} // namespace detail

/**
 * @struct CodedError
 *
 * This error type is used when a function can emit several different instances of the
 * same error. This allows the user to either catch the class the code is based on,
 * or the CodedError with the code specified. This means that the user can
 * handle all errors with a similar cause together, or gain more fine-grained control
 * if needed.
 *
 * @versionadded{1.0.0}
 */
template <class ErrorClass, int ErrorCode>
struct CodedError : ErrorClass {
    using ErrorClass::ErrorClass;

    /**
     * Get the error code for this exception
     *
     * @versionadded{1.0.0}
     */
    constexpr int get_code() const { return ErrorCode; }
};

/**
 * @struct rank_error
 *
 * Indicates that the rank of some tensor arguments are not compatible with the given operation.
 *
 * @versionadded{1.1.0}
 */
struct EINSUMS_EXPORT rank_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct dimension_error
 *
 * Indicates that the dimensions of some tensor arguments are not compatible with the given operation.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT dimension_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct tensor_compat_error
 *
 * Indicates that two or more tensors are incompatible to be operated with each other for a reason other
 * than their dimensions.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT tensor_compat_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct num_argument_error
 *
 * Indicates that a function did not receive the correct amount of arguments.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT num_argument_error : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct not_enough_args
 *
 * Indicates that a function did not receive enough arguments. Child of num_argument_error .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT not_enough_args : num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct too_many_args
 *
 * Indicates that a function received too many arguments. Child of num_argument_error .
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT too_many_args : num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct access_denied
 *
 * Indicates that an operation was stopped due to access restrictions, for instance writing to read-only data.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT access_denied : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct todo_error
 *
 * Indicates that a certain code path is not yet finished.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT todo_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct not_implemented
 *
 * Indicates that a certain code path is not implemented.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT not_implemented : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct bad_logic
 *
 * Indicates that an error occurred for some unspecified reason. It means
 * the same as std::logic_error. However, since so many exceptions are derived from
 * std::logic_error, this acts as a way to not break things.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT bad_logic : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct uninitialized_error
 *
 * Indicates that the code is handling data that is uninitialized.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT uninitialized_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct system_error
 *
 * Indicates that an error happened when making a system call.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT system_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct enum_error
 *
 * Indicates that an invalid enum value was passed to a function.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT enum_error : std::domain_error {
    using std::domain_error::domain_error;
};

/**
 * @struct complex_conversion_error
 *
 * Thrown when trying to convert a complex number to a real number. Instead, the input
 * data should be transformed into a real value in a way that makes sense for the operation
 * being performed. This is often either the magnitude or the real part.
 *
 * @versionadded{2.0.0}
 */
struct EINSUMS_EXPORT complex_conversion_error : std::logic_error {
    using std::logic_error::logic_error;
};

#if defined(EINSUMS_COMPUTE_CODE) || defined(DOXYGEN)
/**
 * @struct hip_exception
 *
 * @brief Wraps an hipError_t value as an exception object to be handled by C++'s exception system.
 *
 * @versionadded{1.0.0}
 */
template <hipError_t error>
struct hip_exception : std::exception {
  private:
    std::string message;

  public:
    /**
     * @brief Construct a new HIP exception.
     *
     * @versionadded{1.0.0}
     */
    hip_exception(char const *diagnostic) : message{""} {
        message += diagnostic;
        message += hipGetErrorString(error);
    }

    /**
     * @brief Construct a new HIP exception.
     *
     * @versionadded{1.0.0}
     */
    hip_exception(std::string diagnostic) : message{""} {
        message += diagnostic;
        message += hipGetErrorString(error);
    }

    /**
     * @brief Return the error string.
     *
     * @versionadded{1.0.0}
     */
    char const *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     *
     * @versionadded{1.0.0}
     */
    template <hipError_t other_error>
    bool operator==(hip_exception<other_error> const &other) const {
        return error == other_error;
    }

    /**
     * @brief Equality operator.
     *
     * @versionadded{1.0.0}
     */
    bool operator==(hipError_t other) const { return error == other; }

    /**
     * @brief Inequality operator.
     *
     * @versionadded{1.0.0}
     */
    template <hipError_t other_error>
    bool operator!=(hip_exception<other_error> const &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     *
     * @versionadded{1.0.0}
     */
    bool operator!=(hipError_t other) const { return error != other; }
#    ifndef DOXYGEN
    template <hipError_t error2>
    friend bool operator==(hipError_t, hip_exception<error2> const &);
    template <hipError_t error2>
    friend bool operator!=(hipError_t, hip_exception<error2> const &);
#    endif
};

/**
 * @brief Reverse equality operator.
 *
 * @versionadded{1.0.0}
 */
template <hipError_t error>
bool operator==(hipError_t first, hip_exception<error> const &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 *
 * @versionadded{1.0.0}
 */
template <hipError_t error>
bool operator!=(hipError_t first, hip_exception<error> const &second) {
    return first != error;
}

/**
 * This can be thrown when a HIP function exits successfully. It is normally suppressed, but can
 * be thrown if requested.
 *
 * @versionadded{1.0.0}
 */
using Success = hip_exception<hipSuccess>;

/**
 * Indicates that at least one of the arguments passed to a function is @c NULL .
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidValue = hip_exception<hipErrorInvalidValue>;

/**
 * Indicates that the requested allocation could not be completed due to not having enough
 * free memory to accomodate.
 *
 * @versionadded{1.0.0}
 */
using ErrorOutOfMemory = hip_exception<hipErrorOutOfMemory>;

/**
 * Indicates a memory allocation error. It is superceded by ErrorOutOfMemory.
 *
 * @versionadded{1.0.0}
 */
using ErrorMemoryAllocation = hip_exception<hipErrorMemoryAllocation>;

/**
 * This is thrown when the GPU device has not been initialized, such as by a call to <tt>hipFree(nullptr);</tt>.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotInitialized = hip_exception<hipErrorNotInitialized>;

/**
 * Superceded by ErrorNotInitialized .
 *
 * @versionadded{1.0.0}
 */
using ErrorInitializationError = hip_exception<hipErrorInitializationError>;

/**
 * This is thrown if an operation is attempted after a device has been de-initialized, such as by a call to
 * @c hipDeviceReset .
 *
 * @versionadded{1.0.0}
 */
using ErrorDeinitialized = hip_exception<hipErrorDeinitialized>;

/**
 * This is thrown if an attempt is made to profile an application, but the profiler has been disabled.
 *
 * @versionadded{1.0.0}
 */
using ErrorProfilerDisabled = hip_exception<hipErrorProfilerDisabled>;

/**
 * This is thrown if an attempt is made to profile an application, but the profiler has not yet been initialized.
 *
 * @versionadded{1.0.0}
 */
using ErrorProfilerNotInitialized = hip_exception<hipErrorProfilerNotInitialized>;

/**
 * This is thrown if the code tries to start the profiler, but the profiler is already running.
 *
 * @versionadded{1.0.0}
 */
using ErrorProfilerAlreadyStarted = hip_exception<hipErrorProfilerAlreadyStarted>;

/**
 * This is thrown if the code tries to stop the profiler, but it has already been stopped.
 *
 * @versionadded{1.0.0}
 */
using ErrorProfilerAlreadyStopped = hip_exception<hipErrorProfilerAlreadyStopped>;

/**
 * This is thrown when the device is not configured for a certain operation, such as interprocess communication.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidConfiguration = hip_exception<hipErrorInvalidConfiguration>;

/**
 * Indicates that the pitch of an array is invalid for some reason.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidPitchValue = hip_exception<hipErrorInvalidPitchValue>;

/**
 * Thrown when an operation tries to access a symbol that is not valid for some reason, such as being null,
 * or not being allocated.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidSymbol = hip_exception<hipErrorInvalidSymbol>;

/**
 * Thrown when a function expects a device pointer, but the pointer provided is invalid. A common issue that can cause this is passing a
 * host pointer instead of a device pointer.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidDevicePointer = hip_exception<hipErrorInvalidDevicePointer>;

/**
 * Thrown when a :code:`hipMemcpy` is initiated, but the pointers are not of the right occupancy for the direction specified.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidMemcpyDirection = hip_exception<hipErrorInvalidMemcpyDirection>;

/**
 * This is thrown when an operation is attempted that is not supported by the current device driver.
 *
 * @versionadded{1.0.0}
 */
using ErrorInsufficientDriver = hip_exception<hipErrorInsufficientDriver>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorMissingConfiguration = hip_exception<hipErrorMissingConfiguration>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorPriorLaunchFailure = hip_exception<hipErrorPriorLaunchFailure>;

/**
 * Thrown when an attempt is made to access or modify a device function, but the requested function is invalid for some reason.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidDeviceFunction = hip_exception<hipErrorInvalidDeviceFunction>;

/**
 * This is thrown when no device can be found to run code.
 *
 * @versionadded{1.0.0}
 */
using ErrorNoDevice = hip_exception<hipErrorNoDevice>;

/**
 * This is thrown when trying to access a device with an ID outside of the range of enumerated devices.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidDevice = hip_exception<hipErrorInvalidDevice>;

/**
 * Thrown when a cooperative group is launched with an invalid image. This is likely due to the code being compiled for the wrong
 * architecture.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidImage = hip_exception<hipErrorInvalidImage>;

/**
 * "Produced when input context is invalid" (from @c hip_runtime_api.h ). This is often due to using a handle to something that is not
 * defined in the current context.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidContext = hip_exception<hipErrorInvalidContext>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorContextAlreadyCurrent = hip_exception<hipErrorContextAlreadyCurrent>;

/**
 * Thrown when an attempt to map some portion of memory into some virtual address space fails.
 *
 * @versionadded{1.0.0}
 */
using ErrorMapFailed = hip_exception<hipErrorMapFailed>;

/**
 * Superceded by ErrorMapFailed .
 *
 * @versionadded{1.0.0}
 */
using ErrorMapBufferObjectFailed = hip_exception<hipErrorMapBufferObjectFailed>;

/**
 * Thrown when an attempt to unmap some portion of virtual memory fails. For instance, if it has already been unmapped, this may be thrown.
 *
 * @versionadded{1.0.0}
 */
using ErrorUnmapFailed = hip_exception<hipErrorUnmapFailed>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorArrayIsMapped = hip_exception<hipErrorArrayIsMapped>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorAlreadyMapped = hip_exception<hipErrorAlreadyMapped>;

/**
 * Raised when there are multiple devices on the system, but a binary can not be found for some of them.
 *
 * @versionadded{1.0.0}
 */
using ErrorNoBinaryForGpu = hip_exception<hipErrorNoBinaryForGpu>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorAlreadyAcquired = hip_exception<hipErrorAlreadyAcquired>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotMapped = hip_exception<hipErrorNotMapped>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotMappedAsArray = hip_exception<hipErrorNotMappedAsArray>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotMappedAsPointer = hip_exception<hipErrorNotMappedAsPointer>;

/**
 * Unused as of now. Likely thrown when a response on the PCI bus fails the error correction check in a way that is detectable but not
 * correctable.
 *
 * @versionadded{1.0.0}
 */
using ErrorECCNotCorrectable = hip_exception<hipErrorECCNotCorrectable>;

/**
 * Thrown when trying to get or set a limit that is not supported by the device.
 *
 * @versionadded{1.0.0}
 */
using ErrorUnsupportedLimit = hip_exception<hipErrorUnsupportedLimit>;

/**
 * Thrown when trying to modify a context currently in use. Deprecated on AMD only.
 *
 * @versionadded{1.0.0}
 */
using ErrorContextAlreadyInUse = hip_exception<hipErrorContextAlreadyInUse>;

/**
 * Thrown when trying to access a device on a multi-device system, but the operation is unsupported by either devices or the driver.
 *
 * @versionadded{1.0.0}
 */
using ErrorPeerAccessUnsupported = hip_exception<hipErrorPeerAccessUnsupported>;

/**
 * Unkown as of now. This seems to be unused but not deprecated. There is also a note in <tt>hip_runtime_api.h - In CUDA DRV, it is
 * CUDA_ERROR_PTX</tt>
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidKernelFile = hip_exception<hipErrorInvalidKernelFile>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidGraphicsContext = hip_exception<hipErrorInvalidGraphicsContext>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidSource = hip_exception<hipErrorInvalidSource>;

/**
 * Thrown when trying to load data from a module, but the file can not be found.
 *
 * @versionadded{1.0.0}
 */
using ErrorFileNotFound = hip_exception<hipErrorFileNotFound>;

/**
 * Thrown when trying to load data from a module, but the requested symbol is not found in the module.
 *
 * @versionadded{1.0.0}
 */
using ErrorSharedObjectSymbolNotFound = hip_exception<hipErrorSharedObjectSymbolNotFound>;

/**
 * Thrown when trying to initialize a shared object library, but an error occurs.
 *
 * @versionadded{1.0.0}
 */
using ErrorSharedObjectInitFailed = hip_exception<hipErrorSharedObjectInitFailed>;

/**
 * Thrown when a binary is built for one operating system but being run on a different system.
 *
 * @versionadded{1.0.0}
 */
using ErrorOperatingSystem = hip_exception<hipErrorOperatingSystem>;

/**
 * Thrown when using a handle for an event or stream that is invalid for some reason.
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidHandle = hip_exception<hipErrorInvalidHandle>;

/**
 * Superceded by ErrorInvalidHandle .
 *
 * @versionadded{1.0.0}
 */
using ErrorInvalidResourceHandle = hip_exception<hipErrorInvalidResourceHandle>;

/**
 * Thrown when a resource is in a state that does not support the requested operation.
 *
 * @versionadded{1.0.0}
 */
using ErrorIllegalState = hip_exception<hipErrorIllegalState>;

/**
 * Thrown when trying to find a variable in a module, but the variable can not be found.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotFound = hip_exception<hipErrorNotFound>;

/**
 * Thrown when trying to query an event's properties when those properties haven't been computed yet. For instance, trying to find the
 * elapsed time, but the event is still running. This is not actually an error, according to the HIP documentation.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotReady = hip_exception<hipErrorNotReady>;

/**
 * Unknown as of now. This seems to be unused but not deprecated.
 *
 * @versionadded{1.0.0}
 */
using ErrorIllegalAddress = hip_exception<hipErrorIllegalAddress>;

/**
 * This is thrown when trying to launch a cooperative kernel, but there are not enough resources available for the kernel to run.
 *
 * @versionadded{1.0.0}
 */
using ErrorLaunchOutOfResources = hip_exception<hipErrorLaunchOutOfResources>;

/**
 * This is thrown when trying to launch a cooperative kernel, but one of the devices times out.
 *
 * @versionadded{1.0.0}
 */
using ErrorLaunchTimeOut = hip_exception<hipErrorLaunchTimeOut>;

/**
 * Thrown when trying to enable peer access on the current device, but it has already been enabled.
 *
 * @versionadded{1.0.0}
 */
using ErrorPeerAccessAlreadyEnabled = hip_exception<hipErrorPeerAccessAlreadyEnabled>;

/**
 * Thrown when trying to access a peer device, but peer access has not been enabled.
 *
 * @versionadded{1.0.0}
 */
using ErrorPeerAccessNotEnabled = hip_exception<hipErrorPeerAccessNotEnabled>;

/**
 * Not well documented. Thrown when trying to set flags for how the device should behave while the device is running a process that would be
 * affected by those changes.
 *
 * @versionadded{1.0.0}
 */
using ErrorSetOnActiveProcess = hip_exception<hipErrorSetOnActiveProcess>;

/**
 * Thrown when trying to destroy a context, but the context has already been destroyed.
 *
 * @versionadded{1.0.0}
 */
using ErrorContextIsDestroyed = hip_exception<hipErrorContextIsDestroyed>;

/**
 * Thrown when an internal assertion fails.
 *
 * @versionadded{1.0.0}
 */
using ErrorAssert = hip_exception<hipErrorAssert>;

/**
 * Thrown when trying to register host memory, but that memory has already been registered.
 *
 * @versionadded{1.0.0}
 */
using ErrorHostMemoryAlreadyRegistered = hip_exception<hipErrorHostMemoryAlreadyRegistered>;

/**
 * Thrown when trying to unregister host memory, but that memory has not been registered or is already unregistered.
 *
 * @versionadded{1.0.0}
 */
using ErrorHostMemoryNotRegistered = hip_exception<hipErrorHostMemoryNotRegistered>;

/**
 * Thrown when something goes wrong when launching a kernel.
 *
 * @versionadded{1.0.0}
 */
using ErrorLaunchFailure = hip_exception<hipErrorLaunchFailure>;

/**
 * Thrown when trying to launch a cooperative kernel with too many blocks.
 *
 * @versionadded{1.0.0}
 */
using ErrorCooperativeLaunchTooLarge = hip_exception<hipErrorCooperativeLaunchTooLarge>;

/**
 * Thrown when the HIP API is not supported.
 *
 * @versionadded{1.0.0}
 */
using ErrorNotSupported = hip_exception<hipErrorNotSupported>;

/**
 * Thrown when trying to perform an operation that is not allowed when a stream is capturing.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureUnsupported = hip_exception<hipErrorStreamCaptureUnsupported>;

/**
 * Thrown when a previous error has invalidated a stream capture.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureInvalidated = hip_exception<hipErrorStreamCaptureInvalidated>;

/**
 * Thrown when an operation would have merged two independent stream captures.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureMerge = hip_exception<hipErrorStreamCaptureMerge>;

/**
 * Thrown when a stream tries to use a capture that was created for another stream.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureUnmatched = hip_exception<hipErrorStreamCaptureUnmatched>;

/**
 * Indicates that a stream capture was forked and never rejoined.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureUnjoined = hip_exception<hipErrorStreamCaptureUnjoined>;

/**
 * Thrown when a cross-stream dependency would have been created.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureIsolation = hip_exception<hipErrorStreamCaptureIsolation>;

/**
 * Thrown when an operation would have caused a disallowed implicit dependency in a capture.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureImplicit = hip_exception<hipErrorStreamCaptureImplicit>;

/**
 * Indicates that an operation was not permitted on an event in a stream that is actively capturing.
 *
 * @versionadded{1.0.0}
 */
using ErrorCapturedEvent = hip_exception<hipErrorCapturedEvent>;

/**
 * A capture operation was initiated on a thread that does not have access to the stream.
 *
 * @versionadded{1.0.0}
 */
using ErrorStreamCaptureWrongThread = hip_exception<hipErrorStreamCaptureWrongThread>;

/**
 * This is thrown when a graph update would have violated certain constraints.
 *
 * @versionadded{1.0.0}
 */
using ErrorGraphExecUpdateFailure = hip_exception<hipErrorGraphExecUpdateFailure>;

/**
 * Thrown when an unknown error has occurred.
 *
 * @versionadded{1.0.0}
 */
using ErrorUnknown = hip_exception<hipErrorUnknown>;

/**
 * An internal memory call produced an error. Not seen on production systems.
 *
 * @versionadded{1.0.0}
 */
using ErrorRuntimeMemory = hip_exception<hipErrorRuntimeMemory>;

/**
 * An internal call that is not a memory call produced an error. Not seen on production systems.
 *
 * @versionadded{1.0.0}
 */
using ErrorRuntimeOther = hip_exception<hipErrorRuntimeOther>;

/**
 * Placeholder error for future expansion.
 *
 * @versionadded{1.0.0}
 */
using ErrorTbd = hip_exception<hipErrorTbd>;

/**
 * @struct hipblas_exception
 * @brief Wraps hipBLAS status codes into an exception.
 *
 * Wraps hipBLAS status codes so that they can be thrown and caught. There is one code for each named hipBLAS status code.
 *
 * @tparam error The status code handled by this exception.
 *
 * @versionadded{1.0.0}
 */
template <hipblasStatus_t error>
struct EINSUMS_EXPORT hipblas_exception : std::exception {
  private:
    std::string message;

  public:
    /**
     * @brief Construct a new hipblas_exception.
     *
     * @versionadded{1.0.0}
     */
    hipblas_exception(char const *diagnostic) : message{""} {
        message += diagnostic;
        message += hipblasStatusToString(error);
    }

    hipblas_exception(std::string diagnostic) : message{""} {
        message += diagnostic;
        message += hipblasStatusToString(error);
    }

    /**
     * @brief Return the error string corresponding to the status code.
     *
     * @versionadded{1.0.0}
     */
    char const *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     *
     * @versionadded{1.0.0}
     */
    template <hipblasStatus_t other_error>
    bool operator==(hipblas_exception<other_error> const &other) const {
        return error == other_error;
    }

    /**
     * @brief Equality operator.
     *
     * @versionadded{1.0.0}
     */
    bool operator==(hipblasStatus_t other) const { return error == other; }

    /**
     * @brief Inequality operator.
     *
     * @versionadded{1.0.0}
     */
    template <hipblasStatus_t other_error>
    bool operator!=(hipblas_exception<other_error> const &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     *
     * @versionadded{1.0.0}
     */
    bool operator!=(hipblasStatus_t other) const { return error != other; }

    template <hipblasStatus_t error2>
    friend bool operator==(hipblasStatus_t, hipblas_exception<error2> const &);
    template <hipblasStatus_t error2>
    friend bool operator!=(hipblasStatus_t, hipblas_exception<error2> const &);
};

/**
 * @brief Reverse equality operator.
 *
 * @versionadded{1.0.0}
 */
template <hipblasStatus_t error>
bool operator==(hipblasStatus_t first, hipblas_exception<error> const &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 *
 * @versionadded{1.0.0}
 */
template <hipblasStatus_t error>
bool operator!=(hipblasStatus_t first, hipblas_exception<error> const &second) {
    return first != error;
}

/**
 * @typedef blasSuccess
 *
 * @brief Represents a success.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasNotInitialized
 *
 * @brief Indicates that the hipBLAS environment has not been properly initialized.
 *
 * Indicates that the hipBLAS environment has not been properly initialized.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasAllocFailed
 *
 * @brief Internal error indicating that resource allocation failed.
 * Internal error indicating that resource allocation failed. Ensure there is enough memory for work arrays,
 * and that any work arrays passed have enough memory.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasInvalidValue
 *
 * @brief Indicates that an unsupported numerical value was passed to a function.
 * Indicates that an unsupported numerical value was passed to a function. An example could be passing an incompatible leading
 * dimension to an array argument.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasMappingError
 *
 * @brief Indicates that access to the GPU memory space failed.
 * Indicates that access to the GPU memory space failed. This may be caused by passing a host pointer to a GPU function,
 * unmapping or unpinning host memory before calling or during a call, or deallocating a pointer while the GPU is still
 * processing the data.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasExecutionFailed
 *
 * @brief Indicates that the GPU program failed to execute.
 * Indicates that the GPU program failed to execute. This could have many causes.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasInternalError
 *
 * @brief Indicates that an unspecified internal error has occurred.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasNotSupported
 *
 * @brief Indicates that the requested operation is not supported.
 * Indicates that the requested operation is not supported. For instance, calling hipblasXscalBatched with an NVidia card.
 * The batched functions are only supported by AMD cards and rocBLAS, not cuBLAS.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasArchMismatch
 *
 * @brief Indicates that the code was compiled for a different architecture than is present.
 * Indicates that the code was compiled for a different architecture than is present. Recompile the code for
 * your device to fix this.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasHandleIsNullptr
 *
 * @brief Indicates that the handle passed is a null pointer.
 * Indicates that the handle passed is a null pointer.
 * If you are calling the hipBLAS functions directly, create a new handle using
 * hipblasCreate and use this in subsequent calls.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasInvalidEnum
 *
 * @brief Indicates that an enum value was passed that was not expected.
 * Indicates that an enum value was passed that was not expected. For instance, passing a value other than
 * HIPBLAS_OP_N (111), HIPBLAS_OP_T (112), or HIPBLAS_OP_C (113) to a transpose argument.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef blasUnknown
 *
 * @brief Indicates an unsupported status code.
 * Indicates an unsupported status code was thrown by the backend. It is also thrown if an unsupported
 * status code is passed to hipblas_catch .
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverSuccess
 *
 * @brief Represents a success.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverNotInitialized
 *
 * @brief Indicates that the hipSolver environment has not been properly initialized.
 *
 * Indicates that the hipBLAS environment has not been properly initialized.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverAllocFailed
 *
 * @brief Internal error indicating that resource allocation failed.
 * Internal error indicating that resource allocation failed. Ensure there is enough memory for work arrays,
 * and that any work arrays passed have enough memory.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverInvalidValue
 *
 * @brief Indicates that an unsupported numerical value was passed to a function.
 * Indicates that an unsupported numerical value was passed to a function. An example could be passing an incompatible leading
 * dimension to an array argument.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverMappingError
 *
 * @brief Indicates that access to the GPU memory space failed.
 * Indicates that access to the GPU memory space failed. This may be caused by passing a host pointer to a GPU function,
 * unmapping or unpinning host memory before calling or during a call, or deallocating a pointer while the GPU is still
 * processing the data.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverExecutionFailed
 *
 * @brief Indicates that the GPU program failed to execute.
 * Indicates that the GPU program failed to execute. This could have many causes.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverInternalError
 *
 * @brief Indicates that an unspecified internal error has occurred.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverFuncNotSupported
 *
 * @brief Indicates that the requested operation is not supported.
 * Indicates that the requested operation is not supported. For instance, using hipsolverRfBatchSolve on an AMD
 * device, which is currently not supported.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverArchMismatch
 *
 * @brief Indicates that the code was compiled for a different architecture than is present.
 * Indicates that the code was compiled for a different architecture than is present. Recompile the code for
 * your device to fix this.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverHandleIsNullptr
 *
 * @brief Indicates that the handle passed is a null pointer.
 * Indicates that the handle passed is a null pointer.
 * If you are calling the hipSolver functions directly, create a new handle using
 * hipsolverCreate and use this in subsequent calls.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverInvalidEnum
 *
 * @brief Indicates that an enum value was passed that was not expected.
 * Indicates that an enum value was passed that was not expected. For instance, passing a value other than
 * HIPSOLVER_OP_N (111), HIPSOLVER_OP_T (112), or HIPSOLVER_OP_C (113) to a transpose argument.
 *
 * @versionadded{1.0.0}
 */

/**
 * @typedef solverUnknown
 *
 * @brief Indicates an unsupported status code.
 * Indicates an unsupported status code was thrown by the backend. It is also thrown if an unsupported
 * status code is passed to hipsolver_catch .
 *
 * @versionadded{1.0.0}
 */

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
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT EINSUMS_HOST char const *hipsolverStatusToString(hipsolverStatus_t status);

/**
 * @struct hipsolver_exception
 *
 * @brief Wraps hipSolver status codes into an exception.
 *
 * Wraps hipSolver status codes into an exception which allows them to be thrown and caught.
 *
 * @tparam error The status code wrapped by the object.
 *
 * @versionadded{1.0.0}
 */
template <hipsolverStatus_t error>
struct EINSUMS_EXPORT hipsolver_exception : std::exception {
  private:
    std::string message;

  public:
    /**
     * Construct a new exception.
     *
     * @versionadded{1.0.0}
     */
    hipsolver_exception(char const *diagnostic) : message{""} {
        message += diagnostic;
        message += hipsolverStatusToString(error);
    }

    /**
     * Construct a new exception.
     *
     * @versionadded{1.0.0}
     */
    hipsolver_exception(std::string diagnostic) : message{""} {
        message += diagnostic;
        message += hipsolverStatusToString(error);
    }

    /**
     * Return the error string.
     *
     * @versionadded{1.0.0}
     */
    char const *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * Equality operator.
     *
     * @versionadded{1.0.0}
     */
    template <hipsolverStatus_t other_error>
    bool operator==(hipsolver_exception<other_error> const &other) const {
        return error == other_error;
    }

    /**
     * Equality operator.
     *
     * @versionadded{1.0.0}
     */
    bool operator==(hipsolverStatus_t other) const { return error == other; }

    /**
     * Inequality operator.
     *
     * @versionadded{1.0.0}
     */
    template <hipsolverStatus_t other_error>
    bool operator!=(hipsolver_exception<other_error> const &other) const {
        return error != other_error;
    }

    /**
     * Inequality operator.
     *
     * @versionadded{1.0.0}
     */
    bool operator!=(hipsolverStatus_t other) const { return error != other; }

#    ifndef DOXYGEN
    template <hipsolverStatus_t error2>
    friend bool operator==(hipsolverStatus_t, hipsolver_exception<error2> const &);
    template <hipsolverStatus_t error2>
    friend bool operator!=(hipsolverStatus_t, hipsolver_exception<error> const &);
#    endif
};

/**
 * Reverse equality operator.
 *
 * @versionadded{1.0.0}
 */
template <hipsolverStatus_t error>
bool operator==(hipsolverStatus_t first, hipsolver_exception<error> const &second) {
    return first == error;
}

/**
 * Reverse inequality operator.
 *
 * @versionadded{1.0.0}
 */
template <hipsolverStatus_t error>
bool operator!=(hipsolverStatus_t first, hipsolver_exception<error> const &second) {
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

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void __hipblas_catch__(hipblasStatus_t status, char const *func_call, char const *fname, char const *diagnostic,
                                                   char const *funcname, bool throw_success = false);

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void __hipsolver_catch__(hipsolverStatus_t status, char const *func_call, char const *fname,
                                                     char const *diagnostic, char const *funcname, bool throw_success = false);

/**
 * Wraps up an HIP function to catch any error codes. If the function does not return
 * hipSuccess, then an exception will be thrown
 *
 * @versionadded{1.0.0}
 */
EINSUMS_HOST EINSUMS_EXPORT void __hip_catch__(hipError_t condition, char const *func_call, char const *fname, char const *diagnostic,
                                               char const *funcname, bool throw_success = false);

#    define hip_catch_STR1(x) #x
#    define hip_catch_STR(x)  hip_catch_STR1(x)
/**
 * @def hip_catch
 *
 * Macro for creating more detailed diagnostic messages when HIP functions fail. Can only take one or two arguments.
 *
 * @versionadded{1.0.0}
 */
#    define hip_catch(condition, ...)                                                                                                      \
        __hip_catch__((condition), "hip_catch" hip_catch_STR((condition)) ";", __FILE__, ":" hip_catch_STR(__LINE__) ":\nIn function: ",   \
                      std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

/**
 * @def hipblas_catch
 *
 * Macro for creating more detailed diagnostic messages when hipBLAS functions fail. Can only take one or two arguments.
 *
 * @versionadded{1.0.0}
 */
#    define hipblas_catch(condition, ...)                                                                                                  \
        __hipblas_catch__(                                                                                                                 \
            (condition), "hipblas_catch" hip_catch_STR((condition)) ";", __FILE__,                                                         \
            ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

/**
 * @def hipsolver_catch
 *
 * Macro for creating more detailed diagnostic messages when hipSolver functions fail. Can only take one or two arguments.
 *
 * @versionadded{1.0.0}
 */
#    define hipsolver_catch(condition, ...)                                                                                                \
        __hipsolver_catch__(                                                                                                               \
            (condition), "hipsolver_catch" hip_catch_STR((condition)) ";", __FILE__,                                                       \
            ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

#endif

} // namespace einsums