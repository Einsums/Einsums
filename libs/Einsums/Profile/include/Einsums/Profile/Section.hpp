
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
 * @versionadded{1.0.0}
 *
 */
struct EINSUMS_EXPORT Section {
#ifdef DOXYGEN
    /**
     * The underlying implementation of the section class.
     *
     * @versionadded{1.0.0}
     */
    struct Impl {
        /**
         * The section name.
         *
         * @versionadded{1.0.0}
         */
        std::string name;

        /**
         * Whether the timer was pushed or not.
         *
         * @versionadded{1.0.0}
         */
        bool push_timer;
    };
#else
    struct Impl;
#endif

    /**
     * @brief Construct a new Section object
     *
     * @param[in] name Name of the section to be timed. If VTune is available then \p name becomes the label in VTune.
     * @param[in] pushTimer Enable Einsums timing mechanism for this section. Default is true.
     *
     * @versionadded{1.0.0}
     */
    explicit Section(std::string const &name, bool pushTimer = true);

    /**
     * @brief Construct a new Section object
     *
     * @param[in] name Name of the section to be timed.
     * @param[in] domain If VTune is available then this is the label used in VTune.
     * @param[in] pushTimer Enable Einsums timing mechanism for this section. Default is true.
     *
     * @versionadded{1.0.0}
     */
    Section(std::string const &name, std::string const &domain, bool pushTimer = true);

    /**
     * @brief Destroy the Section object
     *
     * @versionadded{1.0.0}
     */
    ~Section();

    /**
     * @brief Manually stop the section.
     *
     * If for some reason you need to prematurely end a timing section then use this function.
     * This is automatically called by the destructor when Section goes out of scope.
     *
     *
     * @versionadded{1.0.0}
     */
    void end();

  private:
    void begin();

    /**
     * The underlying implementation of the section class.
     *
     * @versionadded{1.0.0}
     */
    std::unique_ptr<Impl> _impl;
};

} // namespace einsums