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

class PyEinsumsException(BaseException) :
    def __init__(self, *args) :
        super().__init__(*args)
        

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

        for i in range(len(self.__strides)):
            out[i], hold = divmod(hold, self.__strides[i])

        if self.__reverse:
            self.__curr_index -= 1
        else:
            self.__curr_index += 1

        return tuple(out)

__singles = [np.float32]
__doubles = [float, np.float64]
__complex_singles = [np.complex64]
__complex_doubles = [complex, np.complex128]

def create_tensor(*args, dtype=float) :
    if dtype in __singles :
        return core.RuntimeTensorF(*args)
    if dtype in __doubles :
        return core.RuntimeTensorD(*args)
    if dtype in __complex_singles :
        return core.RuntimeTensorC(*args)
    if dtype in __complex_doubles :
        return core.RuntimeTensorZ(*args)
    raise PyEinsumsException(f"Can not create tensor with data type {dtype}!")

def create_random_tensor(name: str, dims: list[int], dtype=float):
    """
    Creates a random tensor with the given name and dimensions.
    """
    out = create_tensor(name, dims, dtype=dtype)

    for ind in TensorIndices(out):
        if dtype in __complex_singles or dtype in __complex_doubles :
            out[ind] = random.random() + 1j * random.random()
        else :
            out[ind] = random.random()

    return out
