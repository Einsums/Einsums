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

/// @todo This class can be freely changed. It is provided as a starting point for your convenience. If not needed, it may be removed.

class EINSUMS_EXPORT Einsums_GPUMemory_vars final : public design_pats::Lockable<std::recursive_mutex> {
    EINSUMS_SINGLETON_DEF(Einsums_GPUMemory_vars)

  public:
    // Put module-global variables here.

    /**
     * @brief Observer to watch for buffer size updates.
     */
    static void update_max_size(config_mapping_type<std::string> const &options);

    /**
     * @brief Resets the current number of allocated bytes to zero.
     */
    void reset_curr_size();

    /**
     * @brief Checks to see if the number of bytes can be allocated.
     *
     * If the number of bytes can not be allocated, returns false. If the number of bytes can be allocated,
     * returns true and updates the current number of allocated bytes.
     */
    bool try_allocate(size_t bytes);

    void deallocate(size_t bytes);

    size_t get_max_size() const;

  private:
    explicit Einsums_GPUMemory_vars() = default;

    size_t max_size_;
    size_t curr_size_;
};
} // namespace detail
} // namespace gpu
} // namespace einsums