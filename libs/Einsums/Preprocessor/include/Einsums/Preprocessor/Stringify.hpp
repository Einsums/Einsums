//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * Converts its argument into a string.
 *
 * @param a The value to turn into a string.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_PP_STRINGIFY(a)  EINSUMS_PP_STRINGIFY2(a)

/**
 * Converts its argument into a string. Second level because of macro weirdness.
 *
 * @param a The value to turn into a string.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_PP_STRINGIFY2(a) #a
