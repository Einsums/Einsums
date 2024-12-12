#include <Einsums/Errors/Error.hpp>

#include <source_location>
#include <fmt/format.h>

EINSUMS_EXPORT std::string einsums::detail::make_error_message(char const *type_name, char const *str, std::source_location const &location) {
    return fmt::format("{}:{}:{}:\nIn {}\n{}: {}", location.file_name(), location.line(), location.column(), location.function_name(),
                       type_name, str);
}

EINSUMS_EXPORT std::string einsums::detail::make_error_message(char const *type_name, std::string const &str, std::source_location const &location) {
    return fmt::format("{}:{}:{}:\nIn {}\n{}: {}", location.file_name(), location.line(), location.column(), location.function_name(),
                       type_name, str);
}