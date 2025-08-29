//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * Concatenates its arguments.
 *
 * @param a
 * @param b The arguments to concatenate.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_PP_CAT(a, b)   EINSUMS_PP_CAT_I(a, b)

/**
 * Concatenates its arguments. Inner level because of macro weirdness.
 *
 * @param a
 * @param b The arguments to concatenate.
 *
 * @versionadded{1.0.0}
 */
#define EINSUMS_PP_CAT_I(a, b) a##b
