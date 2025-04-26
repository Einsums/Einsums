//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <memory>
#include <string>
#include <vector>

namespace einsums::profile::detail {

struct PerformanceCounter {
    virtual ~PerformanceCounter() = default;

    /// The number of events the performance counter will be tracking
    [[nodiscard]] virtual int nevents() = 0;

    /// The names of the events being tracked
    [[nodiscard]] virtual std::vector<std::string> event_names() = 0;

    /// Start capturing performance data
    /// @param s pre-allocated vector of length nevents()
    virtual void start(std::vector<uint64_t> &s) = 0;

    /// Stop capturing and compute delta
    /// @param e pre-allocated vector of length nevents()
    virtual void stop(std::vector<uint64_t> &e) = 0;

    /// Return performance counter results
    /// @param s the data from start()
    /// @param e the data from end(). on return, the difference is stored here
    virtual void delta(std::vector<uint64_t> const &s, std::vector<uint64_t> &e) const = 0;

    /// Factory function to construct the appropriate implementation at runtime
    static std::unique_ptr<PerformanceCounter> create();
};

} // namespace einsums::profile::detail
