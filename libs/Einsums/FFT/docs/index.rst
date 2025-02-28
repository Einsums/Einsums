..
    Copyright (c) The Einsums Developers. All rights reserved.
    Licensed under the MIT License. See LICENSE.txt in the project root for license information.

.. _modules_Einsums_FFT:

===
FFT
===

This module contains wrappers for performing the fast Fourier transform.

See the :ref:`API reference <modules_Einsums_FFT_api>` of this module for more
details.

Public API
----------

.. cpp:function:: void einsums::fft::fft(Tensor<float, 1> const &a, Tensor<std::complex<float>, 2> *result)
.. cpp:function:: void einsums::fft::fft(Tensor<double, 1> const &a, Tensor<std::complex<double>, 2> *result)
.. cpp:function:: void einsums::fft::fft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 2> *result)
.. cpp:function:: void einsums::fft::fft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 2> *result)

    Performs the fast Fourier transform on the input tensor using the linked FFT library.

    :param a[in]: The input tensor.
    :param result[out]: The output tensor.

.. cpp:function:: void einsums::fft::ifft(Tensor<std::complex<float>, 1> const &a, Tensor<float, 2> *result)
.. cpp:function:: void einsums::fft::ifft(Tensor<std::complex<double>, 1> const &a, Tensor<double, 2> *result)
.. cpp:function:: void einsums::fft::ifft(Tensor<std::complex<float>, 1> const &a, Tensor<std::complex<float>, 2> *result)
.. cpp:function:: void einsums::fft::ifft(Tensor<std::complex<double>, 1> const &a, Tensor<std::complex<double>, 2> *result)

    Performs the inverse fast Fourier transform on the input tensor using the linked FFT library.

    :param a[in]: The input tensor.
    :param result[out]: The output tensor.

.. cpp:function:: Tensor<double, 1> fftfreq(int n, double d)

    Gets the frequency corresponding to each position in the frequency tensor.

    :param n: The number of positions in the tensor.
    :param d: The scale factor for the frequency.
    :return: A tensor that contains the frequency at each position in a frequency tensor of the same size.