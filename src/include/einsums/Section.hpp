#pragma once

#include <memory>
#include <string>

struct Section {
    struct Impl;

    Section(const std::string &name, bool pushTimer = true);
    Section(const std::string &name, const std::string &domain, bool pushTimer = true);

    ~Section();

    void end();

  private:
    void begin();

    std::unique_ptr<Impl> _impl;
};
