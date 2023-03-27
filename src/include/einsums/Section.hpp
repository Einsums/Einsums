#pragma once

#include "einsums/_Export.hpp"

#include <memory>
#include <string>

struct EINSUMS_EXPORT Section {
    struct Impl;

    Section(const std::string &name, bool pushTimer = true);
    Section(const std::string &name, const std::string &domain, bool pushTimer = true);

    ~Section();

    void end();

  private:
    void begin();

    std::unique_ptr<Impl> _impl;
};

// Use of LabeledSection requires fmt/format.h to be included and the use of
// (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined in _Common.hpp
#define LabeledSection1(x) Section _section(fmt::format("{}::{} {}", detail::s_Namespace, __func__, x))
#define LabeledSection0()  Section _section(fmt::format("{}::{}", detail::s_Namespace, __func__))
