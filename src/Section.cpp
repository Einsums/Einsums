#include "einsums/Section.hpp"

#include "einsums/Print.hpp"
#include "einsums/Timer.hpp"

#if defined(HAVE_ITTNOTIFY)
#include <ittnotify.h>

__itt_domain *global_domain = __itt_domain_create("Einsums");
#endif

struct Section::Impl {
    std::string name;
#if defined(HAVE_ITTNOTIFY)
    __itt_domain *domain;
    __itt_string_handle *section;
#endif
};

Section::Section(const std::string &name) : _impl{new Section::Impl} {
    _impl->name = name;

#if defined(HAVE_ITTNOTIFY)
    _impl->domain = global_domain;
    _impl->section = __itt_string_handle_create(name.c_str());
#endif

    // println("Entering section: {}", _impl->name);
    // print::indent();
    einsums::timer::push(_impl->name);

#if defined(HAVE_ITTNOTIFY)
    __itt_task_begin(_impl->domain, __itt_null, __itt_null, _impl->section);
#endif
}

Section::Section(const std::string &name, const std::string &domain) : _impl{new Section::Impl} {
    _impl->name = name;

#if defined(HAVE_ITTNOTIFY)
    _impl->domain = __itt_domain_create(domain.c_str());
    _impl->section = __itt_string_handle_create(name.c_str());
#endif

    einsums::timer::push(_impl->name);

#if defined(HAVE_ITTNOTIFY)
    __itt_task_begin(_impl->domain, __itt_null, __itt_null, _impl->section);
#endif
}

Section::~Section() {
#if defined(HAVE_ITTNOTIFY)
    __itt_task_end(_impl->domain);
#endif
    einsums::timer::pop();
}

Frame::Frame() {
#if defined(HAVE_ITTNOTIFY)
    __itt_frame_begin_v3(global_domain, nullptr);
#endif
}

Frame::~Frame() {
#if defined(HAVE_ITTNOTIFY)
    __itt_frame_end_v3(global_domain, nullptr);
#endif
}