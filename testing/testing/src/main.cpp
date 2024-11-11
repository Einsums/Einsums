//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#define CATCH_CONFIG_RUNNER

#include <catch2/catch_all.hpp>

int main(int argc, char **argv) {
    Catch::StringMaker<float>::precision  = 10;
    Catch::StringMaker<double>::precision = 17;

    Catch::Session session;
    session.applyCommandLine(argc, argv);
    int result = session.run();
    return result;
}
