//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <fmt/format.h>

#include <source_location>
#include <string>
#include <system_error>

namespace einsums {

enum class Error : std::uint16_t {
    /// The operation was successful
    success = 0,
    /// The operation failed, but not in an expected way
    no_success,
    unknown_error,
    bad_parameter,
    kernel_error,
    out_of_memory,
    invalid_status,
    tensors_incompatible,
    disk_error,
    assertion_failure,
    unhandled_exception,
    lock_error,

    last_error,

    system_error_flag = 0x4000L,
    error_upper_bound = 0x7fffL
};

namespace detail {
extern char const *const error_names[];

std::string make_error_message(char const *type_name, char const *str, std::source_location const &location);
std::string make_error_message(char const *type_name, std::string const &str, std::source_location const &location);

} // namespace detail

class EinsumsException {
  public:
    EinsumsException(char const *what) : what_{what} {}
    EinsumsException(std::string const &what) : what_{what} {}
    EinsumsException(const EinsumsException &other) = default;
    virtual ~EinsumsException() = default;

    std::string what() const { return this->what_; }

  protected:
    std::string what_;
};

template <Error Code>
class ErrorCode : public EinsumsException {
  public:
    using EinsumsException::EinsumsException;

    ~ErrorCode() = default;

    constexpr inline Error get_code() const { return Code; }
};

// Generate the exceptions.
namespace error {

using success = ErrorCode<Error::success>;
using no_success = ErrorCode<Error::no_success>;
using unknown_error = ErrorCode<Error::unknown_error>;
using bad_parameter = ErrorCode<Error::bad_parameter>;
using kernel_error = ErrorCode<Error::kernel_error>;
using out_of_memory = ErrorCode<Error::out_of_memory>;
using invalid_status = ErrorCode<Error::invalid_status>;
using tensors_incompatible = ErrorCode<Error::tensors_incompatible>;
using disk_error = ErrorCode<Error::disk_error>;
using assertion_failure = ErrorCode<Error::assertion_failure>;
using unhandled_exception = ErrorCode<Error::unhandled_exception>;
using lock_error = ErrorCode<Error::lock_error>;

}

} // namespace einsums