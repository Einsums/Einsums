#include <Einsums/Errors/Error.hpp>

#include <fmt/format.h>

#include <source_location>

bool einsums::detail::__is_library_initialized = false;

void einsums::error::initialize() {
    einsums::detail::__is_library_initialized = true;
}

EINSUMS_EXPORT std::string einsums::detail::make_error_message(char const *type_name, char const *str,
                                                               std::source_location const &location) {
    if (!einsums::detail::__is_library_initialized) {
        return fmt::format("{}:{}:{}:\nIn {}\n{}: {}\nLibrary is not initialized. Try initializing first to see if the error goes away.",
                           location.file_name(), location.line(), location.column(), location.function_name(), type_name, str);
    }
    return fmt::format("{}:{}:{}:\nIn {}\n{}: {}", location.file_name(), location.line(), location.column(), location.function_name(),
                       type_name, str);
}

EINSUMS_EXPORT std::string einsums::detail::make_error_message(char const *type_name, std::string const &str,
                                                               std::source_location const &location) {
    return einsums::detail::make_error_message(type_name, str.c_str(), location);
}