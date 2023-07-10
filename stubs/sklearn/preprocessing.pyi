# pylint: skip-file

from typing import TypeVarTuple
from numpy import ndarray

Shape = TypeVarTuple("Shape")

def minmax_scale(
    X: ndarray[float, *Shape], axis: int = 0, feature_range: tuple[float, float] = (0, 1)
) -> ndarray[float, *Shape]: ...
