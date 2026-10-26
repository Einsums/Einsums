//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/Profile.hpp>

namespace einsums::profile {
#ifdef EINSUMS_HAVE_PROFILER

AggNode::AggNode(std::string const &n) : name_(n) {
    if (name_.length() != 0) {
        std::fprintf(stderr, "Creating AggNode: %s\n", name_.c_str());
    }
}

AggNode::~AggNode() {
    if (name_.length() != 0) {
        std::fprintf(stderr, "Finalizing AggNode: %s\n", name_.c_str());
    } else {
        std::fprintf(stderr, "Finalizing AggNode.\n");
    }
}

std::string const &AggNode::name() const noexcept {
    return name_;
}

std::string &AggNode::name() noexcept {
    return name_;
}

std::string const &AggNode::file() const noexcept {
    return file_;
}

std::string &AggNode::file() noexcept {
    return file_;
}

int AggNode::line() const noexcept {
    return line_;
}

std::string const &AggNode::function() const noexcept {
    return function_;
}

std::string &AggNode::function() noexcept {
    return function_;
}

uint64_t AggNode::call_count() const noexcept {
    return call_count_;
}

ns const &AggNode::total_exclusive() const noexcept {
    return total_exclusive_;
}

int64_t AggNode::total_exclusive_mean() const noexcept {
    return total_exclusive_mean_;
}

int64_t AggNode::total_exclusive_M2() const noexcept {
    return total_exclusive_M2_;
}

ns const &AggNode::exclusive_min() const noexcept {
    return exclusive_min_;
}

ns const &AggNode::exclusive_max() const noexcept {
    return exclusive_max_;
}

std::map<std::string, uint64_t> const &AggNode::counters_total() const noexcept {
    return counters_total_;
}

std::map<std::string, uint64_t> const &AggNode::counters_min() const noexcept {
    return counters_total_;
}

std::map<std::string, uint64_t> const &AggNode::counters_max() const noexcept {
    return counters_total_;
}

std::shared_ptr<AggNode> AggNode::find_or_add(std::string const &name) {
    auto it = children_.find(name);

    if (it == children_.end()) {
        children_[name] = std::make_shared<AggNode>(name);
        it              = children_.find(name);
    }

    return it->second;
}

void AggNode::set_name(std::string const &name) {
    name_ = name;
}

void AggNode::set_file(std::string const &fname) {
    file_ = fname;
}

void AggNode::set_line(int line) noexcept {
    line_ = line;
}

void AggNode::set_function(std::string const &func) {
    function_ = func;
}

void AggNode::inc_call_count() noexcept {
    call_count_++;
}

void AggNode::set_call_count(uint64_t count) noexcept {
    call_count_ = count;
}

void AggNode::inc_exclusive(ns const &time) {
    total_exclusive_ += time;

    int64_t const delta = time.count() - total_exclusive_mean_;
    total_exclusive_mean_ += delta / static_cast<int64_t>(call_count_);

    int64_t const delta2 = time.count() - total_exclusive_mean_;
    total_exclusive_M2_ += delta * delta2;

    if (time < exclusive_min_) {
        exclusive_min_ = time;
    }

    if (time > exclusive_max_) {
        exclusive_max_ = time;
    }
}

InsertionOrderedMap<std::string, std::shared_ptr<AggNode>> const &AggNode::children() const noexcept {
    return children_;
}

#endif
} // namespace einsums::profile