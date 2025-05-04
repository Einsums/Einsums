//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Profile/Detail/PerformanceCounter.hpp>

#include <linux/perf_event.h>
#include <string>
#include <vector>

namespace einsums::profile::detail {
struct PerformanceCounterLinux : PerformanceCounter {
    explicit PerformanceCounterLinux();
    ~PerformanceCounterLinux() override;

    int                      nevents() const override;
    std::vector<std::string> event_names() const override;

    void                                      capture(std::vector<uint64_t> &s) override;
    void                                      delta(std::vector<uint64_t> const &s, std::vector<uint64_t> &e) const override;

    std::unordered_map<std::string, uint64_t> to_event_map(std::vector<uint64_t> const &d) const override;

  private:
    struct PerfFd {
        int  fd;
        bool multiplexed;
    };

    std::vector<PerfFd>      _fds;
    std::vector<std::string> _names;

    struct ReadData {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    };

    static int perf_event_open(perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags);
};
} // namespace einsums::profile::detail