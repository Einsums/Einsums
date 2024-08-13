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
    EinsumsException(const char *location, const char *function, const char *message) : _what{""} {
        _what += location;
        _what += ": In function ";
        _what += function;
        _what += ": ";
        _what += message;
    }

    EinsumsException(const char *location, const char *function, std::string message) : _what{""} {
        _what += location;
        _what += ": In function ";
        _what += function;
        _what += ": ";
        _what += message;
    }

    const char *what() const noexcept override {
        return _what.c_str();
    }

    EinsumsException &operator=(const EinsumsException &other) noexcept {
        _what = other._what;
        return *this;
    }
};

}



#define __EINSUMS_EXCEPTION_STR1__(x) #x
#define __EINSUMS_EXCEPTION_STR__(x) __EINSUMS_EXCEPTION_STR1__(x)
#define EINSUMSEXCEPTION(what) einsums::EinsumsException(einsums::detail::anonymize(__FILE__).c_str(), std::source_location::current().function_name(), (what))