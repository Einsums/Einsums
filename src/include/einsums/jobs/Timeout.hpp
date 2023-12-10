/**
 * @file Timeout.hpp
 *
 * Contains class data for the timeout exception class.
 */

#pragma once

#include "einsums/_Common.hpp"

#include <stdexcept>

BEGIN_EINSUMS_NAMESPACE_HPP(einsums::jobs)

/**
 * @struct timeout
 *
 * Exception thrown when a waiting thread times out.
 */
struct timeout : public std::exception {
  private:
    static constexpr char message[] = "Timeout";

  public:
    /// Constructor.
    timeout() noexcept : std::exception() {}

    /// Copy constructor.
    timeout(const timeout &other) noexcept = default;

    /// Destructor.
    ~timeout() = default;

    /// Returns the message.
    const char *what() noexcept { return timeout::message; }
};

END_EINSUMS_NAMESPACE_HPP(einsums::jobs)