//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Utilities/Random.hpp>

#include <chrono>
#include <memory>
#include <random>

namespace einsums {

std::default_random_engine &random_engine() {
    thread_local std::unique_ptr<std::default_random_engine> instance{nullptr};

    if (!instance) {
        instance = std::make_unique<std::default_random_engine>(std::chrono::system_clock::now().time_since_epoch().count());
    }

    return *instance;
}

} // namespace einsums
