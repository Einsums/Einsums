..
    ----------------------------------------------------------------------------------------------
     Copyright (c) The Einsums Developers. All rights reserved.
     Licensed under the MIT License. See LICENSE.txt in the project root for license information.
    ----------------------------------------------------------------------------------------------

.. _einsums.core.section :

*************************
Einsums Profiler Sections
*************************

.. sectionauthor:: Connor Briggs

.. codeauthor:: Connor Briggs

This class is an advanced topic. It is used internally in :py:deco:`einsums.utils.labeled_section`.

.. py:currentmodule:: einsums.core

.. py:class:: Section

    Represents a section in the profiler report.

    .. versionadded:: 1.1.0

    .. py:method:: __init__(name: str, [domain: str], [push_timer: bool = True])

        Create a new section with the given name and domain.

        :param name: The name of the timer.
        :param domain: If VTune is available, then this label will be used for VTune.
        :param push_timer: Enable timing for this section.

        .. versionadded:: 1.1.0

    .. py:method:: end()

        End timing early.

        .. versionadded:: 1.1.0