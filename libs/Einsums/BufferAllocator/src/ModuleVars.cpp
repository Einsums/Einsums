//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/Logging.hpp>

namespace einsums::detail {

EINSUMS_SINGLETON_IMPL(Einsums_BufferAllocator_vars)

void Einsums_BufferAllocator_vars::update_max_size(config_mapping_type<std::string> const &options) {
    std::perror("Updating buffer size.\n");
    auto       &singleton = get_singleton();
    auto        lock      = std::lock_guard(singleton);
    auto const &value     = options.at("buffer-size");

    singleton.max_size_ = string_util::memory_string(value);
}

size_t Einsums_BufferAllocator_vars::get_max_size() const {
    return max_size_;
}

bool Einsums_BufferAllocator_vars::request_bytes(size_t bytes) {
    auto lock = std::lock_guard(lock_);

    if(bytes + curr_size_ > max_size_) {
        return false;
    }

    curr_size_ += bytes;
    return true;
}

void Einsums_BufferAllocator_vars::release_bytes(size_t bytes) {
    auto lock = std::lock_guard(lock_);

    if(bytes > curr_size_) {
        curr_size_ = 0;
    } else {
        curr_size_ -= bytes;
    }
}

size_t Einsums_BufferAllocator_vars::get_available() const {
    if(curr_size_ > max_size_) {
        return 0;
    }
    return max_size_ - curr_size_;
}


} // namespace einsums::detail