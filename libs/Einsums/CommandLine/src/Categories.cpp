//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>

namespace einsums::cl {

ExclusiveCategory::ExclusiveCategory() {
    Registry::get_singleton().add_exclusion(this);
}

ExclusiveCategory::~ExclusiveCategory() {
    std::fprintf(stderr, "Finalizing ExclusiveCategory.\n");
}

bool ExclusiveCategory::verify_exclusions() const {
    bool found_one = false;

    for (auto const *opt : options) {
        if (opt->is_from_cli() || opt->is_from_config() || opt->num_occurrences() > 0) {
            if (found_one) {
                return false;
            }
            found_one = true;
        }
    }

    return true;
}

void ExclusiveCategory::add_option(OptionBase *option) {
    options.push_back(option);
}

} // namespace einsums::cl