# pylint: skip-file

from typing import TypeVar

from ._base import LinearModel

NFeatures = TypeVar("NFeatures")

class LinearRegression(LinearModel[NFeatures]):
    def __init__(self, fit_intercept: bool = True) -> None: ...
