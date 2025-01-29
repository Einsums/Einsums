//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Profile/Section.hpp>
#include <Einsums/Profile/Timer.hpp>

// TODO: Connect to ITTNOTIFY
namespace einsums {

struct Section::Impl {
    std::string name;
    bool        push_timer;
};

Section::Section(std::string const &name, bool pushTimer) : _impl{new Impl} {
    _impl->name       = name;
    _impl->push_timer = pushTimer;

    // #if defined(EINSUMS_HAVE_ITTNOTIFY)
    // _impl->domain  = global_domain;
    // _impl->section = __itt_string_handle_create(name.c_str());
    // #endif

    begin();
}

Section::Section(std::string const &name, std::string const &domain, bool pushTimer) : _impl{new Impl} {
    _impl->name       = name;
    _impl->push_timer = pushTimer;

    // #if defined(HAVE_ITTNOTIFY)
    // _impl->domain  = __itt_domain_create(domain.c_str());
    // _impl->section = __itt_string_handle_create(name.c_str());
    // #endif

    begin();
}

Section::~Section() {
    end();
}

void Section::begin() {
    if (_impl->push_timer)
        einsums::profile::push(_impl->name);

    // #if defined(HAVE_ITTNOTIFY)
    // __itt_task_begin(_impl->domain, __itt_null, __itt_null, _impl->section);
    // #endif
}

void Section::end() {
    if (_impl) {
        // #if defined(HAVE_ITTNOTIFY)
        // __itt_task_end(_impl->domain);
        // #endif
        if (_impl->push_timer)
            einsums::profile::pop();
    }

    _impl = nullptr;
}
} // namespace einsums