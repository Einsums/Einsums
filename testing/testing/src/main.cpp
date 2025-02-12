//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Runtime.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <functional>


#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

int einsums_main(int argc, char *const *const argv) {
    Catch::Session session;
    session.applyCommandLine(argc, argv);

    Catch::StringMaker<float>::precision  = std::numeric_limits<float>::digits10;
    Catch::StringMaker<double>::precision = std::numeric_limits<double>::digits10;
    auto seed                             = session.config().rngSeed();

    einsums::seed_random(seed);

    int result = session.run();
    einsums::finalize();

    return result;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
