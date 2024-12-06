#include <Einsums/Errors/Error.hpp>

#include <source_location>
#include <type_traits>

using namespace einsums;

std::string einsums::detail::make_error_message(char const *type_name, char const *str, std::source_location const &location) {
    return fmt::format("{}:{}:{}:{}:\n{}: {}", location.file_name(), location.function_name(), location.line(), location.column(),
                       type_name, str);
}

std::string einsums::detail::make_error_message(char const *type_name, std::string const &str, std::source_location const &location) {
    return fmt::format("{}:{}:{}:{}:\n{}: {}", location.file_name(), location.function_name(), location.line(), location.column(),
                       type_name, str);
}