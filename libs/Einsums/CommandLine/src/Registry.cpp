//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/CommandLine.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

namespace einsums::cl {

EINSUMS_SINGLETON_DEF(Registry);

void Registry::add_option(OptionBase *o) {
    options.push_back(o);
}

void Registry::add_category(OptionCategory *c) {
    categories.push_back(c);
}

void Registry::add_exclusion(ExclusiveCategory *c) {
    exclusions.push_back(c);
}

std::list<OptionBase *> const &Registry::get_options() const {
    return options;
}

std::list<OptionCategory *> const &Registry::get_categories() const {
    return categories;
}

std::list<ExclusiveCategory *> const &Registry::get_exclusions() const {
    return exclusions;
}

void Registry::clear_for_tests() {
    options.clear();
    categories.clear();
    exclusions.clear();
}
}
