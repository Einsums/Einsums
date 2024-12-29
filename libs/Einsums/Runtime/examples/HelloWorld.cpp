//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>

int einsums_main() {
    einsums::println("Hello world!");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
    return einsums::init(einsums_main, argc, argv);
}