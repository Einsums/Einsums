#include "einsums/Section.hpp"

#include "einsums/Print.hpp"
#include "einsums/STL.hpp"
#include "einsums/Timer.hpp"

#include <cstddef>

#if defined(HAVE_ITTNOTIFY)
#    include <ittnotify.h>

__itt_domain *global_domain = __itt_domain_create("Einsums");
#endif

struct Section::Impl {
    std::string name;
    bool        push_timer;
#if defined(HAVE_ITTNOTIFY)
    __itt_domain        *domain;
    __itt_string_handle *section;
#endif
};

Section::Section(const std::string &name, bool pushTimer) : _impl{new Section::Impl} {
    _impl->name       = einsums::trim_copy(name);
    _impl->push_timer = pushTimer;

#if defined(HAVE_ITTNOTIFY)
    _impl->domain  = global_domain;
    _impl->section = __itt_string_handle_create(name.c_str());
#endif

    begin();
}

Section::Section(const std::string &name, const std::string &domain, bool pushTimer) : _impl{new Section::Impl} {
    _impl->name       = einsums::trim_copy(name);
    _impl->push_timer = pushTimer;

#if defined(HAVE_ITTNOTIFY)
    _impl->domain  = __itt_domain_create(domain.c_str());
    _impl->section = __itt_string_handle_create(name.c_str());
#endif

    begin();
}

Section::~Section() {
    end();
}

void Section::begin() {
    if (_impl->push_timer)
        einsums::timer::push(_impl->name);

#if defined(HAVE_ITTNOTIFY)
    __itt_task_begin(_impl->domain, __itt_null, __itt_null, _impl->section);
#endif
}

void Section::end() {
    if (_impl) {
#if defined(HAVE_ITTNOTIFY)
        __itt_task_end(_impl->domain);
#endif
        if (_impl->push_timer)
            einsums::timer::pop();
    }

    _impl = nullptr;
}