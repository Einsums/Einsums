# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""
utils.py
--------

Various utilities for Einsums in Python.
"""

from __future__ import annotations

import typing
import random

from einsums import core

import numpy as np


def enumerate_many(*args, start=0):
    """
    Enumerate only takes one iterable argument, but it is sometimes useful to
    be able to enumerate multiple lists at once. This allows you to do that.
    If the iterables are of different lengths, then once one ends, it will emit
    :py:code:`None` in its variables. Once they all end, this will raise
    :py:code:`StopIteraton`.

    :param args: The iterators to iterate over.
    """
    index = 0
    stop_cond = [False for arg in args]

    # Skip the first several indices. Basically copying the code below, but removing the yield.
    for i in range(start):
        while not all(stop_cond):
            out = tuple(index)
            for index, arg in enumerate(args):
                if stop_cond[index]:
                    out = out + tuple(None)
                else:
                    try:
                        out = out + tuple(next(arg))
                    except StopIteration:
                        stop_cond[index] = True
                        out = out + tuple(None)
            if all(stop_cond):
                return

    # Yield the outputs one by one until all the lists are exhausted.
    while not all(stop_cond):
        out = tuple(index)
        for index, arg in enumerate(args):
            if stop_cond[index]:
                out = out + tuple(None)
            else:
                try:
                    out = out + tuple(next(arg))
                except StopIteration:
                    stop_cond[index] = True
                    out = out + tuple(None)
        if all(stop_cond):
            return
        yield out
    return


class TensorIndices:
    """
    TensorIndices
    -------------

    Iterator for moving through the indices of a tensor.
    """

    def __init__(self, other, **kwargs):
        if isinstance(other, (core.RuntimeTensor, core.RuntimeTensorView)):
            reverse = kwargs.reverse if "reverse" in kwargs else False
            self.__curr_index = 0 if not reverse else len(other)
            self.__strides = [0 for i in range(other.rank())]
            self.__reverse = reverse

            size = 1
            for i in reversed(range(other.rank())):
                self.__strides[i] = size
                size *= other.dim(i)
            self.__size = size
        else:
            self.__curr_index = other.__curr_index
            self.__strides = other.__strides
            self.__reverse = other.__reverse
            self.__size = other.__size

    def __iter__(self) -> TensorIndices:
        return self

    def __reversed__(self) -> TensorIndices:
        out = TensorIndices(self)
        out.__reverse = not self.__reverse
        return out

    def __next__(self) -> tuple[int]:
        if self.__curr_index < 0 or self.__curr_index >= self.__size:
            raise StopIteration

        out = [0 for i in range(len(self.__strides))]

        hold = self.__curr_index

        for i, stride in enumerate(self.__strides):
            out[i], hold = divmod(hold, stride)

        if self.__reverse:
            self.__curr_index -= 1
        else:
            self.__curr_index += 1

        return tuple(out)


__singles = [np.float32]
__doubles = [float, np.float64]
__complex_singles = [np.complex64]
__complex_doubles = [complex, np.complex128]


def create_tensor(*args, dtype=float):
    """
    Create a tensor. The arguments will be passed to the constructor.
    The data type can also be specified.

    :param args: The arguments to pass to the constructor.
    :param dtype: The data type to be stored. Can only be single or double precision real or complex floating points.
    """
    if dtype in __singles:
        return core.RuntimeTensorF(*args)
    if dtype in __doubles:
        return core.RuntimeTensorD(*args)
    if dtype in __complex_singles:
        return core.RuntimeTensorC(*args)
    if dtype in __complex_doubles:
        return core.RuntimeTensorZ(*args)
    raise ValueError(f"Can not create tensor with data type {dtype}!")


def tensor_factory(name: str, dims: list[int], dtype=float, method="einsums"):
    if method == "einsums":
        return create_tensor(name, dims, dtype=dtype)
    elif method == "numpy":
        return np.zeros(dims, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )


def remove_complex(dtype):
    """
    Takes a datatype and gives the real equivalent.
    """
    if dtype in __complex_singles:
        return __singles[0]
    if dtype in __complex_doubles:
        return __doubles[0]
    return dtype


def add_complex(dtype):
    """
    Takes a datatype and gives the complex equivalent.
    """
    if dtype in __singles:
        return __complex_singles[0]
    if dtype in __doubles:
        return __complex_doubles[0]
    return dtype


def create_random_numpy_array(dims: list[int], dtype=float):
    rng = np.random.default_rng()

    if dtype in __singles or dtype in __doubles:
        return rng.random(dims, dtype=dtype)
    elif dtype in __complex_singles:
        real_arr = rng.random(dims, dtype=np.float32)
        imag_arr = rng.random(dims, dtype=np.float32)

        return real_arr.astype(np.complex64) + 1.0j * imag_arr.astype(np.complex64)
    elif dtype in __complex_doubles:
        real_arr = rng.random(dims, dtype=np.float64)
        imag_arr = rng.random(dims, dtype=np.float64)

        return real_arr.astype(np.complex128) + 1.0j * imag_arr.astype(np.complex128)
    else:
        raise ValueError("The data type must be real or complex floating point.")


def create_random_tensor(name: str, dims: list[int], dtype=float):
    if dtype in __singles:
        return core.create_random_tensorF(name, dims)
    elif dtype in __doubles:
        return core.create_random_tensorD(name, dims)
    elif dtype in __complex_singles:
        return core.create_random_tensorC(name, dims)
    elif dtype in __complex_doubles:
        return core.create_random_tensorZ(name, dims)
    else:
        raise ValueError(f"Can not create random tensor with data type {dtype}!")


def random_tensor_factory(
    name: str, dims: list[int], dtype: type = float, method: str = "einsums"
):
    if method == "einsums":
        return create_random_tensor(name, dims, dtype)
    elif method == "numpy":
        return create_random_numpy_array(dims, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )


def create_random_definite(name: str, rows: int, mean=1.0, dtype=float):
    if dtype in __singles:
        return core.create_random_definiteF(name, rows, mean)
    elif dtype in __doubles:
        return core.create_random_definiteD(name, rows, mean)
    elif dtype in __complex_singles:
        return core.create_random_definiteC(name, rows, mean)
    elif dtype in __complex_doubles:
        return core.create_random_definiteZ(name, rows, mean)
    else:
        raise ValueError(f"Can not create random tensor with data type {dtype}!")


def create_random_definite_numpy_array(rows: int, mean=1.0, dtype=float):
    return np.array(create_random_definite("", rows, mean, dtype), dtype=dtype)


def random_definite_tensor_factory(
    name: str, rows: int, mean=1.0, dtype: type = float, method: str = "einsums"
):
    if method == "einsums":
        return create_random_definite(name, rows, mean, dtype)
    elif method == "numpy":
        return create_random_definite_numpy_array(rows, mean, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )


def create_random_semidefinite(
    name: str, rows: int, mean=1.0, force_zeros=1, dtype=float
):
    if dtype in __singles:
        return core.create_random_semidefiniteF(name, rows, mean, force_zeros)
    elif dtype in __doubles:
        return core.create_random_semidefiniteD(name, rows, mean, force_zeros)
    elif dtype in __complex_singles:
        return core.create_random_semidefiniteC(name, rows, mean, force_zeros)
    elif dtype in __complex_doubles:
        return core.create_random_semidefiniteZ(name, rows, mean, force_zeros)
    else:
        raise ValueError(f"Can not create random tensor with data type {dtype}!")


def create_random_semidefinite_numpy_array(
    rows: int, mean=1.0, force_zeros=1, dtype=float
):
    return np.array(
        create_random_semidefinite("", rows, mean, force_zeros, dtype), dtype=dtype
    )


def random_semidefinite_tensor_factory(
    name: str,
    rows: int,
    mean=1.0,
    force_zeros=1,
    dtype: type = float,
    method: str = "einsums",
):
    if method == "einsums":
        return create_random_semidefinite(name, rows, mean, force_zeros, dtype)
    elif method == "numpy":
        return create_random_semidefinite_numpy_array(rows, mean, force_zeros, dtype)
    else:
        raise ValueError(
            "Can only produce tensors when the method is 'einsums' or 'numpy'."
        )
