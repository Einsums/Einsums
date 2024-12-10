//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

/**
 * @brief Convenience wrapper to Section.
 *
 * Use of this macro requires that (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined
 * in _Common.hpp is included and used.
 * Constructs a label that includes the encompassing namespace and function names.
 * This macro also includes an extra label that will be appended to the section name.
 */
#define LabeledSection1(x)

/**
 * @brief Convenience wrapper to Section.
 *
 * Use of this macro requires that (BEGIN|END)_EINSUMS_NAMESPACE_(CPP|HPP)() defined
 * in _Common.hpp is included and used.
 * Constructs a label that includes the encompassing namespace and function names.
 */
#define LabeledSection0()
