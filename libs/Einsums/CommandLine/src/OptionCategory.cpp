//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>

namespace einsums::cl {
OptionCategory::OptionCategory(StringRef const &n) : name(n.get_view()) {
    Registry::get_singleton().add_category(this);
}

OptionCategory::~OptionCategory() {
    if (name.length() != 0) {
        std::fprintf(stderr, "Finalizing OptionCategory: %s\n", name.c_str());
    } else {
        std::fprintf(stderr, "Finalizing OptionCategory.\n");
    }
}

std::string const &OptionCategory::get_name() const {
    return name;
}

std::string &OptionCategory::get_name() {
    return name;
}

} // namespace einsums::cl