..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.tensor_algebra :

*****************************
Einsums Python Tensor Algebra
*****************************

.. sectionauthor:: Connor Briggs

This page will discuss functions and classes relating to tensor algebra calls in Python.

.. py:currentmodule:: einsums.core

.. py:function:: compile_plan(C_indices: str, A_indices: str, B_indices: str) -> einsums.core.EinsumsGenericPlan

    Inspects the indices passed in the strings and determines any optimizations that can be done to the subsequent
    operations with this plan. The call will look something like
    :math:`C_{abc\cdots} = \alpha C_{abc\cdots} + \beta A_{def\cdots} B_{ghi\cdots}`, though we
    don't have to specify the prefactors here. Only the indices need to be specified.

    As an example on how to use this, here is a matrix multiplication.

    .. code:: Python

        >>> import einsums
        >>> plan = einsums.core.compile_plan("ij", "ik", "kj")
        >>> A = einsums.util.create_random_tensor("A", [3, 3])
        >>> B = einsums.util.create_random_tensor("B", [3, 3])
        >>> C = einsums.util.create_random_tensor("C", [3, 3])
        >>> plan.execute(0, C, 1, A, B)

    :param C_indices: The indices on the output tensor.
    :param A_indices: The indices on the left input tensor.
    :param B_indices: The indices on the right input tensor.
    :return: A plan which can execute the given tensor operation.

.. py:class:: EinsumsGenericPlan

    This is the base for all plans. It is also the fall-back when a plan that normally could be optimized
    is passed values that it can't handle, such as tensors with different types or tensor views whose smallest
    stride is more than one element (usually).

    .. py:method:: execute(C_prefactor, C, AB_prefactor, A, B) -> None

        Execute the plan. This can work when :code:`C_prefactor` and :code:`AB_prefactor` can be cast to numbers, as well
        as when :code:`A`, :code:`B`, and :code:`C` are either all buffer objects or all :py:class:`einsums.core.GPUView` objects.
        If they are all :py:class:`einsums.core.GPUView` objects, make sure to watch the state of the synchronization of
        the objects, as they may become desynchronized.

.. py:class:: EinsumsDotPlan

    This plan optimizes calls that are equivalent to dot products. At this time, :code:`C` cannot be a scalar value.
    This may change in the future. This does not conjugate :code:`A` or :code:`B`, meaning this is not a true dot product.
    The form of the dot product is this: :math:`C = \alpha C + \beta A_{i} B_{i}`.

    .. py:method:: execute(C_prefactor, C, AB_prefactor, A, B) -> None

        Execute the plan. This can work when :code:`C_prefactor` and :code:`AB_prefactor` can be cast to numbers, as well
        as when :code:`A`, :code:`B`, and :code:`C` are either all buffer objects or all :py:class:`einsums.core.GPUView` objects.
        If they are all :py:class:`einsums.core.GPUView` objects, make sure to watch the state of the synchronization of
        the objects, as they may become desynchronized.

.. py:class:: EinsumsDirectProductPlan

    This plan optimizes calls that are equivalent to direct products. The ideal form of the direct product is this:
    :math:`C_{i} = \alpha C_{i} + \beta A_{i} B_{i}`.

    .. py:method:: execute(C_prefactor, C, AB_prefactor, A, B) -> None

        Execute the plan. This can work when :code:`C_prefactor` and :code:`AB_prefactor` can be cast to numbers, as well
        as when :code:`A`, :code:`B`, and :code:`C` are either all buffer objects or all :py:class:`einsums.core.GPUView` objects.
        If they are all :py:class:`einsums.core.GPUView` objects, make sure to watch the state of the synchronization of
        the objects, as they may become desynchronized.

.. py:class:: EinsumsGerPlan

    This plan optimizes calls that are equivalent to outer products, calling BLAS's :code:`ger` at its core. 
    The ideal form of the direct product is this: :math:`C_{ij} = \alpha C_{ij} + \beta A_{i} B_{j}`.

    .. py:method:: execute(C_prefactor, C, AB_prefactor, A, B) -> None

        Execute the plan. This can work when :code:`C_prefactor` and :code:`AB_prefactor` can be cast to numbers, as well
        as when :code:`A`, :code:`B`, and :code:`C` are either all buffer objects or all :py:class:`einsums.core.GPUView` objects.
        If they are all :py:class:`einsums.core.GPUView` objects, make sure to watch the state of the synchronization of
        the objects, as they may become desynchronized.

.. py:class:: EinsumsGemvPlan

    This plan optimizes calls that are equivalent to matrix-vector products, calling BLAS's :code:`gemv` at its core. 
    The ideal form of the direct product is this: :math:`C_{i} = \alpha C_{i} + \beta A_{ij} B_{j}`.

    .. py:method:: execute(C_prefactor, C, AB_prefactor, A, B) -> None

        Execute the plan. This can work when :code:`C_prefactor` and :code:`AB_prefactor` can be cast to numbers, as well
        as when :code:`A`, :code:`B`, and :code:`C` are either all buffer objects or all :py:class:`einsums.core.GPUView` objects.
        If they are all :py:class:`einsums.core.GPUView` objects, make sure to watch the state of the synchronization of
        the objects, as they may become desynchronized.

.. py:class:: EinsumsGemmPlan

    This plan optimizes calls that are equivalent to matrix products, calling BLAS's :code:`gemm` at its core. 
    The ideal form of the direct product is this: :math:`C_{ij} = \alpha C_{ij} + \beta A_{ik} B_{kj}`.

    .. py:method:: execute(C_prefactor, C, AB_prefactor, A, B) -> None

        Execute the plan. This can work when :code:`C_prefactor` and :code:`AB_prefactor` can be cast to numbers, as well
        as when :code:`A`, :code:`B`, and :code:`C` are either all buffer objects or all :py:class:`einsums.core.GPUView` objects.
        If they are all :py:class:`einsums.core.GPUView` objects, make sure to watch the state of the synchronization of
        the objects, as they may become desynchronized.