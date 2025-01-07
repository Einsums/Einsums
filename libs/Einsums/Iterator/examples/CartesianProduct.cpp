//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Iterator/CartesianProduct.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>

#include <tuple>

using namespace einsums;

int einsums_main() {
    std::vector<int>     a = {1, 2};
    std::array<int, 3>   b = {3, 4, 5};
    std::tuple<int, int> c = {6, 7};

    CartesianProduct product(a, b, c);

    std::size_t total_size = product.end() - product.begin();

    EINSUMS_OMP_PARALLEL_FOR
    for (std::size_t i = 0; i < total_size; ++i) {
        auto iter  = product.begin() + i;
        auto tuple = *iter;

        println("({})", tuple);
    }

    finalize();

    return 0;
}

int main(int argc, char **argv) {
    return einsums::initialize(einsums_main, argc, argv);
}