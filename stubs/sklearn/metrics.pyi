# pylint: skip-file

from typing import TypeVarTuple

from numpy import ndarray

Shape = TypeVarTuple("Shape")

def mean_squared_error(y_true: ndarray[float, *Shape], y_pred: ndarray[float, *Shape]) -> float: ...
