//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/Profile.hpp>

#include <memory>

namespace einsums::profile {

ActiveFrame::ActiveFrame(std::string const &name, TimePoint const &start, std::string const &file, int line, std::string const &function)
    : name_{name}, start_{start}, child_time_{0}, file_{file}, line_{line}, function_{function} {
}

ActiveFrame::~ActiveFrame() {
    if (name_.length() != 0) {
        std::fprintf(stderr, "Finalizing ActiveFrame: %s\n", name_.c_str());
    } else {
        std::fprintf(stderr, "Finalizing ActiveFrame.\n");
    }
}

std::string const &ActiveFrame::name() const noexcept {
    return name_;
}

TimePoint const &ActiveFrame::start() const noexcept {
    return start_;
}

ns const &ActiveFrame::child_time() const noexcept {
    return child_time_;
}

std::string const &ActiveFrame::file() const noexcept {
    return file_;
}

int ActiveFrame::line() const noexcept {
    return line_;
}

std::string const &ActiveFrame::function() const noexcept {
    return function_;
}

std::string &ActiveFrame::name() noexcept {
    return name_;
}

TimePoint &ActiveFrame::start() noexcept {
    return start_;
}

ns &ActiveFrame::child_time() noexcept {
    return child_time_;
}

std::string &ActiveFrame::file() noexcept {
    return file_;
}

int &ActiveFrame::line() noexcept {
    return line_;
}

std::string &ActiveFrame::function() noexcept {
    return function_;
}

void ActiveFrame::inc_child_time(ns const &time) {
    child_time_ += time;
}

ActiveStack::~ActiveStack() {
    std::fprintf(stderr, "Finalizing ActiveStack.\n");
}

ActiveStack &ActiveStack::get_multiton() {
    thread_local std::unique_ptr<ActiveStack> multiton{nullptr};

    if (!multiton) {
        multiton = std::make_unique<ActiveStack>(PrivateConstructorStuff());
    }

    return *multiton;
}

ActiveFrame const &ActiveStack::top() const {
    return stack_.back();
}

ActiveFrame &ActiveStack::top() {
    return stack_.back();
}

bool ActiveStack::empty() const {
    return stack_.empty();
}

size_t ActiveStack::size() const {
    return stack_.size();
}

void ActiveStack::push(ActiveFrame const &value) {
    stack_.push_back(value);
}

void ActiveStack::pop() {
    stack_.pop_back();
}

typename ActiveStack::iterator ActiveStack::begin() {
    return stack_.begin();
}

typename ActiveStack::const_iterator ActiveStack::begin() const {
    return stack_.begin();
}

typename ActiveStack::const_iterator ActiveStack::cbegin() const noexcept {
    return stack_.cbegin();
}

typename ActiveStack::iterator ActiveStack::end() noexcept {
    return stack_.end();
}

typename ActiveStack::const_iterator ActiveStack::end() const noexcept {
    return stack_.end();
}

typename ActiveStack::const_iterator ActiveStack::cend() const noexcept {
    return stack_.cend();
}

#ifdef EINSUMS_WITH_TRACY
TracyStack &TracyStack::get_multiton() {
    thread_local std::unique_ptr<TracyStack> multiton{nullptr};

    if (!multiton) {
        multiton = std::make_unique<TracyStack>(PrivateConstructorStuff());
    }

    return *multiton;
}

std::unique_ptr<tracy::ScopedZone> const &TracyStack::top() const {
    return stack_.top();
}

tracy::ScopedZone &TracyStack::top() {
    return stack_.top();
}

bool TracyStack::empty() const {
    return stack_.empty();
}

size_t TracyStack::size() const {
    return stack_.size();
}

void TracyStack::push(tracy::ScopedZone const &value) {
    stack_.push(value);
}

void TracyStack::pop() {
    stack_.pop();
}
#endif

} // namespace einsums::profile