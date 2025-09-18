//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/StringUtil/MemoryString.hpp>
#include <Einsums/TypeSupport/Lockable.hpp>
#include <Einsums/TypeSupport/Singleton.hpp>

namespace einsums {
namespace gpu {
namespace detail {

/**
 * Contains variables for the buffer allocator, such as how much data can be allocated.
 *
 * @versionadded{1.1.0}
 */
class EINSUMS_EXPORT Einsums_GPUMemory_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_GPUMemory_vars)

  public:
    // Put module-global variables here.

    /**
     * @brief Observer to watch for buffer size updates.
     *
     * @param[in] options The option list to search for updates.
     *
     * @versionadded{1.1.0}
     */
    static void update_max_size(config_mapping_type<std::string> const &options);

    /**
     * @brief Resets the current number of allocated bytes to zero.
     *
     * @versionadded{1.1.0}
     */
    void reset_curr_size();

    /**
     * @brief Checks to see if the number of bytes can be allocated.
     *
     * If the number of bytes can not be allocated, returns false. If the number of bytes can be allocated,
     * returns true and updates the current number of allocated bytes.
     *
     * @param[in] bytes The number of bytes to request.
     *
     * @return True if the requested number of bytes could be allocated.
     *
     * @versionadded{1.1.0}
     */
    bool try_allocate(size_t bytes);

    /**
     * @brief Free up a number of bytes.
     *
     * @param[in] bytes The number of bytes to free.
     *
     * @versionadded{1.1.0}
     */
    void deallocate(size_t bytes);

    /**
     * @brief Gets the maximum number of bytes.
     *
     * @return The maximum number of bytes.
     *
     * @versionadded{1.1.0}
     */
    size_t get_max_size() const;

    /**
     * @brief Gets the number of bytes available.
     *
     * @return The number of bytes available.
     *
     * @versionadded{2.0.0}
     */
    size_t get_available() const;

  private:
    explicit Einsums_GPUMemory_vars() = default;

    /**
     * @brief The maximum number of bytes.
     *
     * @versionadded{1.1.0}
     */
    size_t max_size_;

    /**
     * @brief The current number of allocated bytes.
     */
    size_t curr_size_;
};
} // namespace detail
} // namespace gpu
} // namespace einsums