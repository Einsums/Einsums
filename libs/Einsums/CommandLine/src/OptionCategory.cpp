//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>

namespace einsums::cl {

OptionCategory::OptionCategory(std::string_view const &n) : name(n) {
    Registry::get_singleton().add_category(this);
}

OptionCategory::OptionCategory(std::string const &n) : name(n) {
    Registry::get_singleton().add_category(this);
}

OptionCategory::OptionCategory(char const *n) : name(n) {
    Registry::get_singleton().add_category(this);
}

std::string const &OptionCategory::get_name() const {
    return name;
}

ExlusiveCategory::ExclusiveCategory() { Registry::instance().add_exclusion(this); }

bool ExclusiveCategory::verify_exclusions() const {
    bool found_one = false;

    for (auto const *opt : options) {
        if (opt->seen_cli() || opt->seen_config() || opt->occurrences() > 0) {
            if (found_one) {
                return false;
            }
            found_one = true;
        }
    }

    return true;
}

std::list<OptionBase *> ExclusiveCategory::found_options() {
    std::list<OptionBase *> out;
    for (auto *opt : options) {
        if (opt->seen_cli || opt->seen_config || opt->occurrences > 0) {
            out.push_back(opt);
        }
    }

    return out;
}

std::list<OptionBase *> const &ExcluiveCategory::options() const {
    return options_;
}

void ExclusiveCategory::add_option(OptionBase *option) {
    options_.push_back(option);
}
}
