//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <fmt/format.h>

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
 */
EINSUMS_EXPORT std::string make_error_message(char const *type_name, char const *str, std::source_location const &location);

/// @copydoc make_error_message(char const *,char const *,std::source_location const &)
EINSUMS_EXPORT std::string make_error_message(char const *type_name, std::string const &str, std::source_location const &location);

} // namespace detail

/**
 * @struct CodedError
 *
 * This error type is used when a function can emit several different instances of the
 * same error. This allows the user to either catch the class the code is based on,
 * or the CodedError with the code specified. This means that the user can
 * handle all errors with a similar cause together, or gain more fine-grained control
 * if needed.
 */
template <class ErrorClass, int ErrorCode>
struct CodedError : public ErrorClass {
  public:
    using ErrorClass::ErrorClass;

    /**
     * Get the error code for this exception
     */
    constexpr inline int get_code() const { return ErrorCode; }
};

/**
 * @struct dimension_error
 *
 * Indicates that the dimensions of some tensor arguments are not compatible with the given operation.
 */
struct EINSUMS_EXPORT dimension_error : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct tensor_compat_error
 *
 * Indicates that two or more tensors are incompatible to be operated with each other for a reason other
 * than their dimensions.
 */
struct EINSUMS_EXPORT tensor_compat_error : public std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct num_argument_error
 *
 * Indicates that a function did not receive the correct amount of arguments.
 */
struct EINSUMS_EXPORT num_argument_error : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/**
 * @struct not_enough_args
 *
 * Indicates that a function did not receive enough arguments. Child of num_argument_error .
 */
struct EINSUMS_EXPORT not_enough_args : public num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct too_many_args
 *
 * Indicates that a function received too many arguments. Child of num_argument_error .
 */
struct EINSUMS_EXPORT too_many_args : public num_argument_error {
    using num_argument_error::num_argument_error;
};

/**
 * @struct access_denied
 *
 * Indicates that an operation was stopped due to access restrictions, for instance writing to read-only data.
 */
struct EINSUMS_EXPORT access_denied : public std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct todo_error
 *
 * Indicates that a certain code path is not yet finished.
 */
struct EINSUMS_EXPORT todo_error : public std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct bad_logic
 *
 * Indicates that an error occured for some unspecified reason. It means
 * the same as std::logic_error. However, since so many exceptions are derived from
 * std::logic_error, this acts as a way to not break things.
 */
struct EINSUMS_EXPORT bad_logic : public std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @struct uninitialized_error
 *
 * Indicates that the code is handling data that is uninitialized.
 */
struct EINSUMS_EXPORT uninitialized_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @struct enum_error
 *
 * Indicates that an invalid enum value was passed to a function.
 */
struct EINSUMS_EXPORT enum_error : public std::domain_error {
    using std::domain_error::domain_error;
};

#ifdef EINSUMS_COMPUTE_CODE
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
    hip_exception(char const *diagnostic) : message{""} {
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
    char const *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     */
    template <hipError_t other_error>
    bool operator==(hip_exception<other_error> const &other) const {
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
    bool operator!=(hip_exception<other_error> const &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(hipError_t other) const { return error != other; }

    friend bool operator==(hipError_t, hip_exception<error> const &);
    friend bool operator!=(hipError_t, hip_exception<error> const &);
};

/**
 * @brief Reverse equality operator.
 */
template <hipError_t error>
bool operator==(hipError_t first, hip_exception<error> const &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 */
template <hipError_t error>
bool operator!=(hipError_t first, hip_exception<error> const &second) {
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
struct EINSUMS_EXPORT hipblas_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * @brief Construct a new hipblas_exception.
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
     */
    char const *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * @brief Equality operator.
     */
    template <hipblasStatus_t other_error>
    bool operator==(hipblas_exception<other_error> const &other) const {
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
    bool operator!=(hipblas_exception<other_error> const &other) const {
        return error != other_error;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(hipblasStatus_t other) const { return error != other; }

    friend bool operator==(hipblasStatus_t, hipblas_exception<error> const &);
    friend bool operator!=(hipblasStatus_t, hipblas_exception<error> const &);
};

/**
 * @brief Reverse equality operator.
 */
template <hipblasStatus_t error>
bool operator==(hipblasStatus_t first, hipblas_exception<error> const &second) {
    return first == error;
}

/**
 * @brief Reverse inequality operator.
 */
template <hipblasStatus_t error>
bool operator!=(hipblasStatus_t first, hipblas_exception<error> const &second) {
    return first != error;
}

// Put the status code documentaion in a different header for cleaner code.
#    include "Einsums/Errors/hipblas_status_doc.hpp"

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
EINSUMS_EXPORT EINSUMS_HOST const char *hipsolverStatusToString(hipsolverStatus_t status);

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
struct EINSUMS_EXPORT hipsolver_exception : public std::exception {
  private:
    ::std::string message;

  public:
    /**
     * Construct a new exception.
     */
    hipsolver_exception(char const *diagnostic) : message{""} {
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
    char const *what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override { return message.c_str(); }

    /**
     * Equality operator.
     */
    template <hipsolverStatus_t other_error>
    bool operator==(hipsolver_exception<other_error> const &other) const {
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
    bool operator!=(hipsolver_exception<other_error> const &other) const {
        return error != other_error;
    }

    /**
     * Inquality operator.
     */
    bool operator!=(hipsolverStatus_t other) const { return error != other; }

    friend bool operator==(hipsolverStatus_t, hipsolver_exception<error> const &);
    friend bool operator!=(hipsolverStatus_t, hipsolver_exception<error> const &);
};

/**
 * Reverse equality operator.
 */
template <hipsolverStatus_t error>
bool operator==(hipsolverStatus_t first, hipsolver_exception<error> const &second) {
    return first == error;
}

/**
 * Reverse inequality operator.
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
 */
EINSUMS_HOST EINSUMS_EXPORT void __hipblas_catch__(hipblasStatus_t status, char const *func_call, char const *fname, char const *diagnostic,
                                                   char const *funcname, bool throw_success = false);

/**
 * @brief Takes a status code as an argument and throws the appropriate exception.
 *
 * @param status The status to convert.
 * @param throw_success If true, then an exception will be thrown if a success status is passed. If false, then a success will cause the
 * function to exit quietly.
 */
EINSUMS_HOST EINSUMS_EXPORT void __hipsolver_catch__(hipsolverStatus_t status, char const *func_call, char const *fname,
                                                     char const *diagnostic, char const *funcname, bool throw_success = false);

/**
 * Wraps up an HIP function to catch any error codes. If the function does not return
 * hipSuccess, then an exception will be thrown
 */
EINSUMS_HOST EINSUMS_EXPORT void __hip_catch__(hipError_t condition, char const *func_call, char const *fname, char const *diagnostic,
                                               char const *funcname, bool throw_success = false);

#    define hip_catch_STR1(x) #x
#    define hip_catch_STR(x)  hip_catch_STR1(x)
#    define hip_catch(condition, ...)                                                                                                      \
        __hip_catch__((condition), "hip_catch" hip_catch_STR((condition)) ";", __FILE__, ":" hip_catch_STR(__LINE__) ":\nIn function: ",   \
                      std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

#    define hipblas_catch(condition, ...)                                                                                                  \
        __hipblas_catch__(                                                                                                                 \
            (condition), "hipblas_catch" hip_catch_STR((condition)) ";", __FILE__,                                                         \
            ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

#    define hipsolver_catch(condition, ...)                                                                                                \
        __hipsolver_catch__(                                                                                                               \
            (condition), "hipsolver_catch" hip_catch_STR((condition)) ";", __FILE__,                                                       \
            ":" hip_catch_STR(__LINE__) ":\nIn function: ", std::source_location::current().function_name() __VA_OPT__(, ) __VA_ARGS__)

#endif

} // namespace einsums