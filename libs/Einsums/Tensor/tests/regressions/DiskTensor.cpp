//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/Tensor/DiskTensor.hpp>
#include <Einsums/Tensor/InitModule.hpp>
#include <Einsums/Tensor/ModuleVars.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <H5Tpublic.h>
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

#include <catch2/catch_all.hpp>

using namespace einsums;

TEMPLATE_TEST_CASE("Disk tensor double free", "[disk-tensor]", float, double, std::complex<float>, std::complex<double>) {
    DiskTensor<TestType, 2> A_disk{fmt::format("/test/regression/{}/A", einsums::type_name<TestType>()), 10, 10};
    Tensor<TestType, 2>     A_core{"A_core", 10, 10};
    for (size_t i = 0; i < 10; i++)
        for (size_t j = 0; j < 10; j++)
            A_core(i, j) = (double)(i * 10 + j);
    A_disk(All, All) = A_core;

    REQUIRE_NOTHROW([&]() {
        auto view = A_disk(All, All); // DiskView, _constructed == false initially
        view.fetch();                 // forces _tensor construction, _constructed = true
        view.unget();                 // manually destroys _tensor, _constructed = false
    }());
}