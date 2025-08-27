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

/**
 * Contains variables for the BufferAllocator module.
 *
 * @versionadded{1.1.0}
 */
class EINSUMS_EXPORT Einsums_BufferAllocator_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_BufferAllocator_vars)

  public:
    // Put module-global variables here.

    /**
     * @fn get_singleton()
     *
     * Get the single unique instance of this class.
     *
     * @return The single unique instance of the class.
     *
     * @versionadded{1.1.0}
     */
    #ifdef DOXYGEN
    static Einsums_BufferAllocator_vars &get_singleton();
    #endif

    /**
     * @brief Requests a number of bytes from the counter.
     *
     * @versionadded{1.1.0}
     */
    bool request_bytes(size_t bytes);

    /**
     * @brief Releases a number of bytes counter.
     *
     * @versionadded{1.1.0}
     */
    void release_bytes(size_t bytes);

    /**
     * @brief Update the maximum size of the counter.
     *
     * @versionadded{1.1.0}
     */
    static void update_max_size(config_mapping_type<std::string> const &options);

    /**
     * @brief Get the maximum size of the counter.
     *
     * @versionadded{1.1.0}
     */
    size_t get_max_size() const;

    /**
     * @brief Get available bytes.
     *
     * @versionadded{1.1.0}
     */
    size_t get_available() const;

  private:
    explicit Einsums_BufferAllocator_vars() = default;

    /**
     * The maximum number of bytes available to the buffers.
     *
     * @versionadded{1.1.0}
     */
    size_t max_size_{0};

    /**
     * The current number of bytes allocated by Einsums.
     *
     * @versionadded{1.1.0}
     */
    size_t curr_size_{0};

    /**
     * Mutex for avoiding race conditions.
     *
     * @versionadded{1.1.0}
     */
    std::mutex lock_;
};

} // namespace detail
} // namespace einsums