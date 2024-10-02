"""
utils.py
--------

Various utilities for Einsums in Python.
"""

from __future__ import annotations

import typing

import core
import random


class TensorIndices:
    """
    TensorIndices
    -------------

    Iterator for moving through the indices of a tensor.
    """

    def __init__(self, other, **kwargs):
        if isinstance(other, (core.RuntimeTensor, core.RuntimeTensorView)) :
            reverse = kwargs.reverse if "reverse" in kwargs else False
            self.__curr_index = 0 if not reverse else len(other)
            self.__strides = [0 for i in range(other.rank())]
            self.__reverse = reverse

            size = 1
            for i in reversed(range(other.rank())):
                self.__strides[i] = size
                size *= other.dim(i)
            self.__size = size
        else :
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


def create_random_tensor(name: str, dims: list[int]) -> core.RuntimeTensor :
    """
    Creates a random tensor with the given name and dimensions.
    """
    out = core.RuntimeTensor(name, dims)

    for ind in TensorIndices(out):
        out[ind] = random.random()

    return out
