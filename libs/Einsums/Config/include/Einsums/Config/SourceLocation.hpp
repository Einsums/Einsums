//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
#    include <source_location>

namespace einsums {
using std::source_location;
}
#else

#    include <cstdint>

namespace einsums {
struct source_location {
    EINSUMS_DISABLE_WARNING_PUSH
    EINSUMS_DISABLE_WARNING_PREDEFINED_IDENTIFIER
    static consteval auto current(uint_least32_t line = __LINE__, const char *file_name = __FILE__,
                                  const char *function = __PRETTY_FUNCTION__) noexcept -> source_location {
        return {line, 0, file_name, function};
    }
    EINSUMS_DISABLE_WARNING_POP
    constexpr source_location(uint_least32_t line, uint_least32_t column, char const *file_name, char const *function_name) noexcept
        : _line(line), _column(column), _file_name(file_name), _function_name(function_name) {}

    [[nodiscard]] constexpr auto line() const noexcept -> uint_least32_t { return _line; }
    [[nodiscard]] constexpr auto column() const noexcept -> uint_least32_t { return _column; }
    [[nodiscard]] constexpr auto file_name() const noexcept -> char const * { return _file_name; }
    [[nodiscard]] constexpr auto function_name() const noexcept -> char const * { return _function_name; }

  private:
    uint_least32_t _line;
    uint_least32_t _column;
    char const    *_file_name;
    char const    *_function_name;
};
} // namespace einsums

#endif
