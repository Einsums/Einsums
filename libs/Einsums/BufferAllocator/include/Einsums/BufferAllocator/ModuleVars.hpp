//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

#include <source_location>

namespace einsums {
namespace detail {

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

class EINSUMS_EXPORT Einsums_BufferAllocator_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_BufferAllocator_vars)

  public:
    // Put module-global variables here.

    /**
     * @brief Requests a number of bytes from the counter.
     */
    bool request_bytes(size_t bytes);

    /**
     * @brief Releases a number of bytes counter.
     */
    void release_bytes(size_t bytes);

    /**
     * @brief Update the maximum size of the counter.
     */
    static void update_max_size(config_mapping_type<std::string> const &options);

    /**
     * @brief Get the maximum size of the counter.
     */
    size_t get_max_size() const;

    /**
     * @brief Get available bytes.
     */
    size_t get_available() const;

  private:
    explicit Einsums_BufferAllocator_vars() = default;

    size_t max_size_{0};
    size_t curr_size_{0};

    std::mutex lock_;
};

} // namespace detail
} // namespace einsums