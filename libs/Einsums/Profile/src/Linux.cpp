//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include "Linux.hpp"

#include <Einsums/Logging.hpp>

#include <filesystem>
#include <fstream>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>

namespace einsums::profile::detail {

namespace {
constexpr int kMaxReasonableEvents = 32; // arbitrary upper limit for trial-and-error

bool is_wsl() {
    std::ifstream file("/proc/sys/kernel/osrelease");
    if (!file.is_open()) {
        return false;
    }
    std::string osrelease;
    std::getline(file, osrelease);
    return osrelease.find("Microsoft") != std::string::npos || osrelease.find("microsoft") != std::string::npos;
}

bool is_performance_mode() {
    namespace fs                = std::filesystem;
    std::string const base_path = "/sys/devices/system/cpu/";
    for (auto const &entry : fs::directory_iterator(base_path)) {
        if (!entry.is_directory())
            continue;
        std::string const cpu_path = entry.path();
        if (cpu_path.find("cpu") == std::string::npos)
            continue;
        std::string const governor_file = cpu_path + "/cpufreq/scaling_governor";
        std::ifstream     governor(governor_file);
        if (!governor.is_open())
            continue; // Some "cpu" directories may not have cpufreq (like cpu0, cpu1 ok, cpuX missing)
        std::string mode;
        std::getline(governor, mode);

        if (mode != "performance") {
            return false;
        }
    }

    return true;
}

} // namespace

PerformanceCounterLinux::PerformanceCounterLinux() {
    if (is_wsl()) {
        EINSUMS_LOG_WARN("WSL detected, disabling performance counters");
        return;
    }

    if (!is_performance_mode()) {
        EINSUMS_LOG_WARN("CPU is not set to 'performance' mode. "
                         "Timing measurements may be unreliable due to frequency scaling.\n"
                         "To fix: sudo cpupower frequency-set -g performance");
    }

    perf_event_attr pe;
    pe.size           = sizeof(pe);
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    pe.inherit        = 1;
    pe.read_format    = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;

    std::vector<std::pair<uint32_t, std::string>> requested_events = {{PERF_COUNT_HW_CPU_CYCLES, "cycles"},
                                                                      {PERF_COUNT_HW_INSTRUCTIONS, "instructions"},
                                                                      {PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branches"},
                                                                      {PERF_COUNT_HW_BRANCH_MISSES, "branch-misses"},
                                                                      {PERF_COUNT_HW_CACHE_MISSES, "cache-misses"}};

    for (auto const &[type, name] : requested_events) {
        pe.type   = PERF_TYPE_HARDWARE;
        pe.config = type;

        int fd = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd == -1) {
            if (errno == EINVAL || errno == ENOSPC) {
                EINSUMS_LOG_ERROR("Skipping unsupported or excess event: {}", name);
                continue;
            } else {
                EINSUMS_LOG_ERROR("perf_event_open failed for {}", name);
                continue;
            }
        }

        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

        _fds.push_back({fd, false});
        _names.push_back(name);
    }
}

PerformanceCounterLinux::~PerformanceCounterLinux() {
    for (auto &[fd, _] : _fds)
        close(fd);
}

int PerformanceCounterLinux::nevents() const {
    return static_cast<int>(_fds.size());
}

std::vector<std::string> PerformanceCounterLinux::event_names() const {
    return _names;
}

void PerformanceCounterLinux::start(std::vector<uint64_t> &s) {
    s.resize(_fds.size() * 3);
    for (size_t i = 0; i < _fds.size(); ++i) {
        ReadData data{};
        if (read(_fds[i].fd, &data, sizeof(data)) != sizeof(data)) {
            EINSUMS_LOG_ERROR("Failed to read perf counter");
            continue;
        }
        s[i * 3 + 0] = data.value;
        s[i * 3 + 1] = data.time_enabled;
        s[i * 3 + 2] = data.time_running;
    }
}

void PerformanceCounterLinux::stop(std::vector<uint64_t> &e) {
    e.resize(_fds.size() * 3);
    for (size_t i = 0; i < _fds.size(); ++i) {
        ReadData data{};
        if (read(_fds[i].fd, &data, sizeof(data)) != sizeof(data)) {
            EINSUMS_LOG_ERROR("Failed to read perf counter");
            continue;
        }
        e[i * 3 + 0] = data.value;
        e[i * 3 + 1] = data.time_enabled;
        e[i * 3 + 2] = data.time_running;
    }
}

void PerformanceCounterLinux::delta(std::vector<uint64_t> const &s, std::vector<uint64_t> &e) const {
    for (size_t i = 0; i < _fds.size(); ++i) {
        uint64_t delta_val     = e[i * 3 + 0] - s[i * 3 + 0];
        uint64_t delta_enabled = e[i * 3 + 1] - s[i * 3 + 1];
        uint64_t delta_running = e[i * 3 + 2] - s[i * 3 + 2];

        if (delta_running > 0 && delta_running != delta_enabled) {
            delta_val = delta_val * delta_enabled / delta_running;
        }

        e[i] = delta_val;
    }
    e.resize(_fds.size());
}

int PerformanceCounterLinux::perf_event_open(perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

std::unordered_map<std::string, uint64_t> PerformanceCounterLinux::to_event_map(std::vector<uint64_t> const &d) const {
    std::unordered_map<std::string, uint64_t> result;

    for (int i = 0; i < nevents(); i++) {
        result[_names[i]] = d[i];
    }

    return result;
}

std::unique_ptr<PerformanceCounter> PerformanceCounter::create() {
    return std::make_unique<PerformanceCounterLinux>();
}

} // namespace einsums::profile::detail