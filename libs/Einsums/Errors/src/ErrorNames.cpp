//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Errors/Error.hpp>

#include <fmt/format.h>

namespace einsums::detail {

std::string make_error_message(std::string_view const &type_name, char const *str, std::source_location const &location) {
    // if (!__is_library_initialized) {
    // return fmt::format("{}:{}:{}:\nIn {}\n{}: {}\nLibrary is not initialized. Try initializing first to see if the error goes away.",
    // location.file_name(), location.line(), location.column(), location.function_name(), type_name, str);
    // }
    return fmt::format("{}:{}:{}:\nIn {}\n{}: {}", location.file_name(), location.line(), location.column(), location.function_name(),
                       type_name, str);
}

std::string make_error_message(std::string_view const &type_name, std::string const &str, std::source_location const &location) {
    return make_error_message(type_name, str.c_str(), location);
}
} // namespace einsums::detail