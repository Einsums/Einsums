//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/ExportDefinitions.hpp>

#include <string>
#include <vector>

namespace einsums {
EINSUMS_EXPORT std::string difference(std::string const &st1, std::string const &st2);

EINSUMS_EXPORT std::string reverse(std::string const &str);

template <typename int_type, typename Alloc>
void find_char_with_position(std::string const &needles, std::string const &haystack, std::vector<int_type, Alloc> *out) {
    out->resize(needles.length());

    for (int i = 0; i < needles.size(); i++) {
        out->at(i) = (int_type)haystack.find(needles.at(i));
    }
}

} // namespace einsums