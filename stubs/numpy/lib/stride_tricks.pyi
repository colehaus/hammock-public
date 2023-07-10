# pylint: skip-file

from typing import TypeVar
from numpy import ndarray

X = TypeVar("X")
Dim1 = TypeVar("Dim1")
NWindowShape = TypeVar("NWindowShape", bound=int)

# The output is actually a little shorter than `Dim1`. How much shorter depends on the size of the window.
def sliding_window_view(x: ndarray[X, Dim1], window_shape: NWindowShape) -> ndarray[X, Dim1, NWindowShape]: ...
