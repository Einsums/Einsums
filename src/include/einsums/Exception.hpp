#pragma once

#include "einsums/_Common.hpp"

namespace einsums {

namespace detail {
EINSUMS_EXPORT std::string anonymize(std::string fpath);
}

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
#define EINSUMSEXCEPTION(what)                                                                                                             \
    einsums::EinsumsException(einsums::detail::anonymize(__FILE__).c_str(), __EINSUMS_EXCEPTION_STR__(__LINE__),                           \
                              std::source_location::current().function_name(), (what))