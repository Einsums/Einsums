#pragma once

#include "einsums/_Export.hpp"

#include <source_location>
#include <stdexcept>
#include <string>

namespace einsums {

/**
 * @struct EinsumsException
 *
 * @brief Represents an exception that includes information on where it was thrown from.
 */
struct EinsumsException : std::exception {
  private:
    std::string _what;

  public:
    EinsumsException(const char *file, const char *line, const char *function, const char *message) : _what{""} {
        _what += file;
        _what += ":";
        _what += line;
        _what += ":\nIn function ";
        _what += function;
        _what += ":\n";
        _what += message;
    }

    EinsumsException(const char *file, const char *line, const char *function, std::string message) : _what{""} {
        _what += file;
        _what += ":";
        _what += line;
        _what += ":\nIn function ";
        _what += function;
        _what += ":\n";
        _what += message;
    }

    const char *what() const noexcept override { return _what.c_str(); }

    EinsumsException &operator=(const EinsumsException &other) noexcept {
        _what = other._what;
        return *this;
    }
};

} // namespace einsums

#define __EINSUMS_EXCEPTION_STR1__(x) #x
#define __EINSUMS_EXCEPTION_STR__(x)  __EINSUMS_EXCEPTION_STR1__(x)
/**
 * @def EINSUMSEXCEPTION
 *
 * @brief Create an EinsumsException with the specified message.
 *
 * This creates the exception and initializes it with the file it was thrown from, the line it was
 * thrown from within the file, and the name of the function that threw it. It also adds a user-specified
 * message.
 */
#define EINSUMSEXCEPTION(what)                                                                                                             \
    einsums::EinsumsException(__FILE__, __EINSUMS_EXCEPTION_STR__(__LINE__), std::source_location::current().function_name(), (what))
