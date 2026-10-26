//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Types.hpp>
#include <Einsums/Tensor/Tensor.hpp>

#include <Einsums/Testing.hpp>

TEST_CASE("TensorView creation", "[tensor]") {
    using namespace einsums;
    // With the aid of deduction guides we can choose to not specify the rank on the tensor
    einsums::Tensor     A("A", 3, 3, 3);
    einsums::TensorView viewA(A, einsums::Dim{3, 9});

    // Since we are changing the underlying datatype to float the deduction guides will not work.
    einsums::Tensor     fA("A", 3, 3, 3);
    einsums::TensorView fviewA(fA, einsums::Dim{3, 9});

    for (int i = 0, ijk = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++, ijk++)
                A(i, j, k) = ijk;

    REQUIRE((A.dim(0) == 3 && A.dim(1) == 3 && A.dim(2) == 3));
    REQUIRE((viewA.dim(0) == 3 && viewA.dim(1) == 9));

    if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
        for (int i = 0, ij = 0; i < 3; i++)
            for (int j = 0; j < 9; j++, ij++)
                REQUIRE(viewA(i, j) == A(i, j / 3, j % 3));
    } else {
        for (int i = 0, ij = 0; i < 3; i++)
            for (int j = 0; j < 9; j++, ij++)
                REQUIRE(viewA(i, j) == A(i, j % 3, j / 3));
    }

    double *array = new double[100];

    for (int i = 0; i < 100; i++) {
        array[i] = i;
    }

    // Drop down in scope to make sure the view is deleted before the array it is viewing.
    {
        TensorView<double, 2>       view1{array, Dim<2>{10, 10}}, view2{array, Dim{10, 10}, Stride{10, 1}};
        TensorView<double, 2> const const_view1{(double const *)array, Dim<2>{10, 10}},
            const_view2{(double const *)array, Dim{10, 10}, Stride{10, 1}};

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
                    REQUIRE(view1(j, i) == array[i + j * 10]);
                    REQUIRE(const_view1(j, i) == array[i + j * 10]);
                } else {
                    REQUIRE(view1(i, j) == array[i + j * 10]);
                    REQUIRE(const_view1(i, j) == array[i + j * 10]);
                }
                REQUIRE(view2(i, j) == array[10 * i + j]);
                REQUIRE(const_view2(i, j) == array[10 * i + j]);
            }
        }
    }

    delete[] array;
}