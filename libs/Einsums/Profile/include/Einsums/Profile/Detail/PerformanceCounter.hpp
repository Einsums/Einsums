//------------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//------------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <memory>
#include <string>
#include <vector>

namespace einsums::profile::detail {

struct PerformanceCounter {
    virtual ~PerformanceCounter() = default;

    /// The number of events the performance counter will be tracking
    [[nodiscard]] virtual int nevents() const = 0;

    /// The names of the events being tracked
    [[nodiscard]] virtual std::vector<std::string> event_names() const = 0;

    /// Capture the current state of the counters
    /// @param e pre-allocated vector of length nevents()
    /// @note The data stored in e will not be valid until you call the delta function
    virtual void capture(std::vector<uint64_t> &e) = 0;

    /// Return performance counter results
    /// @param s the data from start()
    /// @param e the data from end(). on return, the difference is stored here
    virtual void delta(std::vector<uint64_t> const &s, std::vector<uint64_t> &e) const = 0;

    /// Return the results in a map. This will add an "event" labeled "time"
    /// which is the result of "cycles" / CPU frequency in nanoseconds.
    /// @param d the data in e from the delta function
    virtual std::unordered_map<std::string, uint64_t> to_event_map(std::vector<uint64_t> const &d) const = 0;

    /// Factory function to construct the appropriate implementation at runtime
    static std::unique_ptr<PerformanceCounter> create();
};

} // namespace einsums::profile::detail
