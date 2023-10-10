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

If you desire to compile from source, see .

If you have Python, you can install Einsums with::

    conda install einsums -C psi4


How to create a Tensor
----------------------

To create an Einsums Tensor, you can use the function :cpp:func:`einsums::create_tensor`.

All you need to do to create a simple tensor is pass a name for the tensor and the
dimensionality of each index.::

    #include <einsums.hpp>

    int main(int, char**) {
        auto A = einsums::create_tensor("A", 2, 2);

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