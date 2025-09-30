//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <random>

namespace einsums {

/**
 * @property random_engine
 *
 * @brief The global random engine for random number generation.
 */
EINSUMS_EXPORT extern std::default_random_engine random_engine;

/**
 * @brief Set the seed of the random number generator.
 *
 * @param seed The new seed for the random number generator.
 */
EINSUMS_EXPORT void seed_random(std::default_random_engine::result_type seed);

} // namespace einsums
