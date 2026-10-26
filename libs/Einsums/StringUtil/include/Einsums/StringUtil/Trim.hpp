//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <string>

namespace einsums::string_util {

/**
 * Trim whitespace from the start of the string.
 *
 * @param[inout] s The string to trim.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *    No longer modifies its argument due to potential incompatibility with Windows build systems.
 *    Now returns a new string instead.
 * @endversion
 */
[[nodiscard]] inline std::string ltrim(std::string const &s) {
    // Find the position of the first non-space character.
    auto start_iter = s.begin();

    while (start_iter != s.end() && std::isspace(*start_iter)) {
        start_iter++;
    }

    return std::string(start_iter, s.end());
}

/**
 * Trim whitespace from the end of the string.
 *
 * @param[inout] s The string to trim.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *    No longer modifies its argument due to potential incompatibility with Windows build systems.
 *    Now returns a new string instead.
 * @endversion
 */
[[nodiscard]] inline std::string rtrim(std::string const &s) {
    auto end_iter = std::prev(s.end());

    while (end_iter != s.begin() && std::isspace(*end_iter)) {
        end_iter--;
    }

    if (end_iter == s.begin() && std::isspace(*end_iter)) {
        return std::string();
    } else {
        return std::string(s.begin(), end_iter + 1);
    }
}

/**
 * Trim whitespace from the start and end of the string.
 *
 * @param[inout] s The string to trim.
 *
 * @versionadded{1.0.0}
 * @versionchangeddesc{2.0.0}
 *    No longer modifies its argument due to potential incompatibility with Windows build systems.
 *    Now returns a new string instead.
 * @endversion
 */
[[nodiscard]] inline std::string trim(std::string const &s) {
    auto start_iter = s.begin();

    while (start_iter != s.end() && std::isspace(*start_iter)) {
        start_iter++;
    }

    auto end_iter = std::prev(s.end());

    while (end_iter != start_iter && std::isspace(*end_iter)) {
        end_iter--;
    }

    if (end_iter == start_iter && std::isspace(*end_iter)) {
        return std::string();
    } else {
        return std::string(start_iter, end_iter + 1);
    }
}

} // namespace einsums::string_util