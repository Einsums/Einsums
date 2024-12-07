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

} // namespace einsums