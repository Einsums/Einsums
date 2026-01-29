//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/GPUMemory/ModuleVars.hpp>
#include <Einsums/Logging.hpp>

namespace einsums::gpu::detail {

EINSUMS_SINGLETON_IMPL(Einsums_GPUMemory_vars)

void Einsums_GPUMemory_vars::update_max_size(config_mapping_type<std::string> const &options) {
    auto       &singleton = get_singleton();
    auto        lock      = std::lock_guard(singleton);
    auto const &value     = options.at("gpu-buffer-size");

    singleton.max_size_ = string_util::memory_string(value);

    if (singleton.max_size_ == 0) {
        hip_catch(hipDeviceGetLimit(&singleton.max_size_, hipLimitMallocHeapSize));
    }

    auto const &work_value    = options.at("gpu-work-size");

    singleton.work_size_ = string_util::memory_string(work_value);

    if(singleton.work_size_ == 0) {
        singleton.work_size_ = 2147483648;
    }

    if(singleton.work_size_ > 2147483648) {
        EINSUMS_LOG_WARN("Due to the HIP interface using 32-bit integers for many size parameters, buffers larger than 2MB are not recommended and will break things.");
    }
}

void Einsums_GPUMemory_vars::reset_curr_size() {
    curr_size_ = 0;
}

bool Einsums_GPUMemory_vars::try_allocate(size_t bytes) {
    auto lock = std::lock_guard(*this);

    if (bytes + curr_size_ > max_size_) {
        return false;
    } else {
        curr_size_ += bytes;
        return true;
    }
}

void Einsums_GPUMemory_vars::deallocate(size_t bytes) {
    auto lock = std::lock_guard(*this);

    if (curr_size_ < bytes) {
        curr_size_ = 0;
    } else {
        curr_size_ -= bytes;
    }
}

size_t Einsums_GPUMemory_vars::get_max_size() const {
    return max_size_;
}

size_t Einsums_GPUMemory_vars::get_available() const {
    return max_size_ - curr_size_;
}

size_t Einsums_GPUMemory_vars::get_work_size() const {
    return work_size_;
}

} // namespace einsums::gpu::detail