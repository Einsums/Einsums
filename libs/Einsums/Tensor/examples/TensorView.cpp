//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#include <Einsums/Print.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/Tensor.hpp>
#include <Einsums/TensorUtilities.hpp>

int einsums_main() {
    using namespace einsums;

    println("TensorView Examples");

    //
    // Example 1
    //
    // Create a lower-order view of a higher-order tensor
    // Equivalent to numpy's reshape
    //
    {
        print::Indent _indent;

        println("Example 1");
        // Create an order-3 tensor
        auto A = create_incremented_tensor("A", 3, 3, 3);
        println(A);

        // View the order-3 tensor as an order-2 tensor.
        TensorView viewA(A, Dim{3, 9});
        println(viewA);
    }

    //
    // Example 2
    //
    // A subview of a tensor using offsets and strides
    //
    {
        print::Indent _indent;
        println("Example 2");

        auto       A = create_incremented_tensor("A", 3, 3);
        TensorView viewA(A, Dim{2, 2}, Offset{1, 1}, Stride{3, 1});

        println(A);
        println(viewA);

        // In this example:
        //   A(1, 1) == viewA(0, 0)
        //   A(1, 2) == viewA(0, 1)
        //   A(2, 1) == viewA(1, 0)
        //   A(2, 2) == viewA(1, 1)
    }

    //
    // Example 3
    //
    // A skewed subview of a tensor using offsets and strides
    //
    {
        print::Indent _indent;
        println("Example 3");

        auto       A = create_incremented_tensor("A", 3, 3);
        TensorView viewA(A, Dim{2, 2}, Offset{0, 0}, Stride{4, 1});

        println(A);
        println(viewA);

        // In this example:
        //   A(0, 0) == viewA(0, 0)
        //   A(0, 1) == viewA(0, 1)
        //   A(1, 1) == viewA(1, 0)
        //   A(1, 2) == viewA(1, 1)
    }

    //
    // Example 4
    //
    // A subview of a tensor using offsets and strides that skips one row and one column.
    // Of a 3x3 matrix the view should be of the corners.
    //
    {
        print::Indent _indent;
        println("Example 4");

        auto       A = create_incremented_tensor("A", 3, 3);
        TensorView viewA(A, Dim{2, 2}, Offset{0, 0}, Stride{6, 2});

        println(A);
        println(viewA);

        // In this example, the view is of the four corners.
        //   A(0, 0) == viewA(0, 0)
        //   A(0, 2) == viewA(0, 1)
        //   A(2, 0) == viewA(1, 0)
        //   A(2, 2) == viewA(1, 1)
    }

    return finalize();
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
