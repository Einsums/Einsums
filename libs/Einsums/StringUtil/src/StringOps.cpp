//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/StringUtil/StringOps.hpp>

#include <string>

std::string einsums::difference(std::string const &st1, std::string const &st2) {
    std::string out = st1;

    for (char i : st2) {
        size_t index = out.find(i);
        if (index < out.size()) {
            out.erase(index);
        }
    }

    return out;
}

std::string einsums::reverse(std::string const &str) {
    return std::string{str.rbegin(), str.rend()};
}