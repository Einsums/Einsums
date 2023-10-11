.. Einsums documentation master file, created by
   sphinx-quickstart on Sat Sep 30 06:26:14 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _einsums_docs_mainpage:

#######################
Einsums's documentation
#######################

.. toctree::
   :maxdepth: 2
   :hidden:

   Getting Started <user/absolute_beginners>
   API Reference <api/library_root>

**Version**: |release|

**Useful links**:
`Source Repository <https://github.com/Einsums/Einsums>`_ |
`Issue Tracker <https://github.com/Einsums/Einsums/issues>`_


Einsums provides compile-time contraction pattern analysis to determine optimal tensor
operation to perform. Einsums is a package for scientific computing in C++. It is a C++
library that provides a multidimensional tensor object and an assortment of functions
for fast operations on tensors, including mathematical, shape manipulation, sorting,
I/O, discrete Fourier transforms, basic linear algebra, and tensor algebra.

As a short example, the following call to :code:`einsum` will optimize at compile-time to a BLAS
dgemm call:

.. code-block::

   using einsums;                        // Provides Tensor and create_random_tensor
   using einsums::tensor_algebra;        // Provides einsum and Indices
   using einsums::tensor_algebra::index; // Provides i, j, k

   Tensor<2> A = create_random_tensor("A", 7, 7);
   Tensor<2> B = create_random_tensor("B", 7, 7);
   Tensor<2> C = create_tensor("C", 7, 7);

   einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);


.. grid:: 2

   .. grid-item-card::
      :img-top: _static/index-images/getting_started.svg

      Getting Started
      ^^^^^^^^^^^^^^^

      New to Einsums? Check out the Absolute Beginner's Guide. It contains
      an introduction to Einsums' main concepts and links to additional
      tutorials.

      +++

      .. button-ref:: user/absolute_beginners
         :expand:
         :color: secondary
         :click-parent:

         To the absolute beginner's guide

   .. grid-item-card::
      :img-top: _static/index-images/user_guide.svg

      User Guide
      ^^^^^^^^^^

      The user guide provides in-depth information on the key concepts of
      Einsums with useful background information and explanation.

      +++

   .. grid-item-card::
      :img-top: _static/index-images/api.svg

      API Reference
      ^^^^^^^^^^^^^

      The reference guide contains a detailed description of the functions,
      modules, and objects included in Einsums. The reference describes how the
      methods work and how to use the parameters. It assumes you have an
      understanding of key concepts.

      +++

   .. grid-item-card::
      :img-top: _static/index-images/contributor.svg

      Contributor's Guide
      ^^^^^^^^^^^^^^^^^^^

      Want to add to the codebase? The contributing guidelines will guide you
      through the process of improving Einsums.

      +++




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
