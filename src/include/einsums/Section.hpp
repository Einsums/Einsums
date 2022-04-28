#pragma once

#include <memory>
#include <string>

struct Section {
    struct Impl;

    Section(const std::string &name);
    Section(const std::string &name, const std::string &domain);

    ~Section();

  private:
    std::unique_ptr<Impl> _impl;
};

struct Frame {
    Frame();
    ~Frame();
};