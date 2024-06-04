.. 
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums_docs_mainpage:

#######################
Einsums's documentation
#######################

.. toctree::
   :maxdepth: 2
   :hidden:

   User Guide <user/index>
   API Reference <reference/index>
   Building from source <building/index>
   Contributor's guide <contrib/index>

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

   using einsums;                        // Provides Tensor, create_tensor, and create_random_tensor
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
         :color: primary
         :click-parent:

         To the beginner's guide

   .. grid-item-card::
      :img-top: _static/index-images/user_guide.svg

      User Guide
      ^^^^^^^^^^

      The user guide provides in-depth information on the key concepts of
      Einsums with useful background information and explanation.

      +++

      .. button-ref:: user/index
         :expand:
         :color: primary
         :click-parent:

         To the user's guide

   .. grid-item-card::
      :img-top: _static/index-images/api.svg

      API Reference
      ^^^^^^^^^^^^^

      The reference guide contains a detailed description of the functions,
      modules, and objects included in Einsums. The reference describes how the
      methods work and how to use the parameters. It assumes you have an
      understanding of key concepts.

      +++

      .. button-ref:: reference/index
         :expand:
         :color: primary
         :click-parent:

         To the API reference guide

   .. grid-item-card::
      :img-top: _static/index-images/contributor.svg

      Contributor's Guide
      ^^^^^^^^^^^^^^^^^^^

      Want to add to the codebase? The contributing guidelines will guide you
      through the process of improving Einsums.

      +++

      .. button-ref:: contrib/index
         :expand:
         :color: primary
         :click-parent:

         To the contributer's guide




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
