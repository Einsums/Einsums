//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <string>

namespace einsums::string_util {

// trim from start (in place)
static inline void ltrim(std::string &s) {
    auto first_nospace = std::find_if(s.cbegin(), s.cend(), [](unsigned char ch) {return !std::isspace(ch);});
    if(first_nospace != s.cend()) {
        s.erase(s.begin(), first_nospace);
    }
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    rtrim(s);
    ltrim(s);
}

// trim from start (copying)
static inline auto ltrim_copy(std::string s) -> std::string {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline auto rtrim_copy(std::string s) -> std::string {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline auto trim_copy(std::string s) -> std::string {
    trim(s);
    return s;
}

} // namespace einsums::string_util