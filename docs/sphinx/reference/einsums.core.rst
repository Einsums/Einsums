..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core :

*******************
Einsums Core Module
*******************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

This module contains many functions and classes implemented in C++.

.. py:currentmodule:: einsums.core

.. py:function:: gpu_enabled() -> bool

    Check whether :code:`einsums` was built with GPU acceleration enabled. 

    :return: :code:`True` if GPU capabilities are enabled.

.. py:function:: initialize()

    Initialize the Einsums library. It does not need to be called directly, and instead
    it is called when you import :code:`einsums`. 

.. py:function:: finalize(arg: bool | str)

    Finalize the Einsums library, optionally printing timing info. See :py:func:`einsums.set_finalize_arg`
    to change how the module calls this function. This does not need to be called directly, as it is
    registered with the :code:`atexit` module. 

.. py:function:: report([output_file: str])

    Print the timing report to standard output. If an output file is provided, print to that file instead.

    :param output_file: The file to optionally print to.

.. py:function:: get_buffer_format(buffer) -> str

    Get the format string for a buffer as seen from C++.

    :param buffer: The buffer object to query.


.. toctree::
    :maxdepth: 3

    einsums.core.tensor_algebra
    einsums.core.gpu_view
    einsums.core.runtimetensor
    einsums.core.runtimetensorview
    einsums.core.testing_utils
    einsums.core.tensoriterator