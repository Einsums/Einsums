
//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <string>

namespace einsums {

/**
 * @brief Convenience class for timing blocks of code.
 *
 * The Section class is a wrapper to the timer system used in Einsums.
 * Additionally, if at configure time VTune is detected then sections
 * are also registered with VTune to aid in analysis.
 *
 * @code
 * void test_code() {
 *   Section timingSection{"test_code"};
 *
 *   // Perform some time consuming action.
 * }
 * @endcode
 *
 */
struct EINSUMS_EXPORT Section {
    struct Impl;

    /**
     * @brief Construct a new Section object
     *
     * @param name Name of the section to be timed. If VTune is available then \p name becomes the label in VTune.
     * @param pushTimer Enable Einsums timing mechanism for this section. Default is true.
     */
    explicit Section(std::string const &name, bool pushTimer = true);

    /**
     * @brief Construct a new Section object
     *
     * @param name Name of the section to be timed.
     * @param domain If VTune is available then this is the label used in VTune.
     * @param pushTimer Enable Einsums timing mechanism for this section. Default is true.
     */
    Section(std::string const &name, std::string const &domain, bool pushTimer = true);

    /**
     * @brief Destroy the Section object
     */
    ~Section();

    /**
     * @brief Manually stop the section.
     *
     * If for some reason you need to prematurely end a timing section then use this function.
     * This is automatically called by the destructor when Section goes out of scope.
     *
     */
    void end();

  private:
    void begin();

    std::unique_ptr<Impl> _impl;
};

} // namespace einsums