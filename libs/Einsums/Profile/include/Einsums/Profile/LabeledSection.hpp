//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Profile/Section.hpp>

#include <fmt/format.h>

/**
 * @brief Convenience wrapper to Section.
 *
 * Constructs a label that includes the encompassing namespace and function names.
 * This macro also includes an extra label that will be appended to the section name.
 */
#define LabeledSection1(x) const Section _section(fmt::format("{} {}", __func__, x))

/**
 * @brief Convenience wrapper to Section.
 *
 * Constructs a label that includes the encompassing namespace and function names.
 */
#define LabeledSection0() const Section _section(fmt::format("{}", __func__))
