
..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_GPUStreams:

==========
GPUStreams
==========

This module contains several interfaces for interacting with GPU streams. Einsums associates
a different stream for each OpenMP thread.

See the :ref:`API reference <modules_Einsums_GPUStreams_api>` of this module for more
details.

----------
Public API
----------

For the most part, the functions in this file are used internally. However, there are a few very useful synchronization
functions that may help users to avoid race conditions.

.. cpp:function:: void stream_wait(bool may_skip = false)

    This function waits for the stream associated with the current thread to finish. The argument should be considered
    an advanced feature.

.. cpp:function:: void all_stream_wait()

    This function waits for all streams associated with this process to finish execution.

.. cpp:macro:: all_streams_wait()

    For those who want a plural form of :cpp:func:`all_stream_wait()`. This macro is simply an alias of that function.

.. cpp:function:: void device_synchronize()

    This function waits for the GPU to reach a synchronization point. It is similar to the :cpp:func:`all_stream_wait` function,
    though it is more general than that.
