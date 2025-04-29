//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Profile/Detail/CPUFrequency.hpp>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#if defined(EINSUMS_APPLE)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(EINSUMS_LINUX)
#    include <unistd.h>
#endif

namespace einsums::profile::detail {

auto cpu_frequency() -> uint64_t {
#if defined(EINSUMS_APPLE)
    uint64_t frequency = 0;
    size_t   size      = sizeof(frequency);
    if (sysctlbyname("hw.cpufrequency", &frequency, &size, nullptr, 0) == 0) {
        return frequency; // In Hz
    } else {
        return 0; // Failed
    }

#elif defined(EINSUMS_LINUX)
    // Linux: try /sys first, then fall back to /proc
    std::ifstream sysfs("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (sysfs.is_open()) {
        uint64_t khz = 0;
        sysfs >> khz;
        sysfs.close();
        if (khz > 0) {
            return khz * 1000;
        }
    }

    // Fall back to /proc/cpuinfo
    std::ifstream proc("/proc/cpuinfo");
    std::string   line;
    while (std::getline(proc, line)) {
        if (line.find("cpu MHz") != std::string::npos) {
            std::size_t colon = line.find(':');
            if (colon != std::string::npos) {
                double mhz = std::atof(line.substr(colon + 1).c_str());
                return static_cast<uint64_t>(mhz * 1'000'000);
            }
        }
    }
    return 0; // failed
#else
    return 0;
#endif
}

} // namespace einsums::profile::detail