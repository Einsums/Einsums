..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

******************************************
Einsums: the absolute basics for beginners
******************************************

Welcome to the absolute beginner's guide to Einsums! If you have comments or
suggestions, please do not hesitate to `reach out <https://github.com/Einsums/Einsums/discussions>`_!

Welcome to Einsums!
-------------------

Einsums is an open-source C++ library. It provides **Tensor**, a homogeneous n-dimensional
tensor object, with methods to efficiently operate on it. Einsums can be used to perform
a wide variety of mathematical operations on tensors.

Installing Einsums
------------------

You have two choices for installing Einsums, compiling from source directly or by using conda.
Compiling from source allows you to enable/disable certain features and to include debugging
options into your code.

If you desire to compile from source, visit
:ref:`Building from source <building-from-source>`.

If you have Python, you can install Einsums with::

    conda install einsums

How to include Einsums
----------------------

To access Einsums within your C++ code, you'll need to know the locations of the headers and library.
If you are using CMake use can use the :code:`find_package` function. Then with the
:code:`target_link_libraries` function you can link against Einsums.

For example, in your CMakeLists.txt file you can have lines similar to the following:

.. parsed-literal::
    find_package(Einsums \ |release| \ CONFIG)

    add_executable(sample main.cpp)
    target_link_libraries(samples Einsums::einsums)

Then in your main.cpp you can have something like

.. code-block:: c++

    #include <einsums.hpp>

    int main() {

        auto A = einsums::create_tensor("A", 3, 3);

        return 0;
    }

Reading the example code
------------------------------

If you are not already comfortable with reading tutorials that contain a lot code,
you might not know how to interpret a code block that looks
like this::

    auto A = einsums::create_random_tensor(6);
    auto B = einsums::Tensor{std::move(A), -1, 6};
    B.dims();  // --> Dims{1, 6 }

If you are not familiar with this style, it's very easy to understand.
If you see do not see ``-->``, you're looking at the **input**, or the code that
you would type. Everything that is comment and has ``-->`` in front of it is potential
**output** or a representation of what you should expect.  The lines with
``-->`` should not be copied into your code and will cause a compile error
if types or pasted into your code.

How to create a Tensor
----------------------

To create an Einsums Tensor, you can use the function :cpp:func:`einsums::create_tensor`.

All you need to do to create a basic tensor is pass a name for the tensor and the
dimensionality of each index.::

    #include <einsums.hpp>

    int main(int, char**) {
        auto A = einsums::create_tensor("A", 2, 2);  // --> einsums::Tensor<2, double>

        return 0;
    }

In this example, we are using the C++ ``auto`` to simplify the type signature. We can
write the data type explicitly if we want to.::

    #include <einsums.hpp>

    int main(int, char**) {
        // Full explicit data type
        einsums::Tensor<2, double> A = einsums::create_tensor("A", 2, 2);

        // The default underlying type of a tensor is `double`
        einsums::Tensor<2> B = einsums::create_tensor("B", 2, 2);

        // Allow the compiler to determine things.
        auto C = einsums::create_tensor("C", 2, 2);

        return 0;
    }

**Specifying your data type**

While the default data type is double-precision floating point (``double``), you
can explicitly specify which data type you want use.::

    auto A = einsums::create_tensor<float>("A", 2, 2);

    auto B = einsums::Tensor<float>{"B", 2, 2};

Einsums also supports the use of complex numbers.::

    auto C = einsums::create_tensor<std::complex<double>>("C", 2, 2);

    auto D = einsums::Tensor<std::complex<float>>{"D", 2, 2};

TODO: Write a more complete guide of creating Tensors in a quickstart guide.

TODO: Adding, removing, and sorting elements
--------------------------------------------

TODO: Shape and size of a Tensor
--------------------------------

TODO: Reshaping a Tensor
------------------------

TODO: Converting a 1D Tensor into a 2D Tensor
---------------------------------------------

TODO: Indexing and slicing
--------------------------

TODO: Basic Tensor operations
-----------------------------

TODO: More useful Tensor operations
-----------------------------------

TODO: Transposing and reshaping a Tensor
----------------------------------------

TODO: