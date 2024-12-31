//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Runtime.hpp>

#include <functional>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

int einsums_main(int argc, char **argv) {
    Catch::Session session;
    session.applyCommandLine(argc, argv);
    return session.run();
}

int main(int argc, char **argv) {
    Catch::StringMaker<float>::precision  = std::numeric_limits<float>::digits10;
    Catch::StringMaker<double>::precision = std::numeric_limits<double>::digits10;

    auto const wrapped = std::bind_front(&einsums_main, argc, argv);

    return einsums::initialize(wrapped, argc, argv);
}
