# pylint: skip-file

from typing import Generic, TypeVar, TypeVarTuple

from numpy import ndarray

Shape = TypeVarTuple("Shape")

NSamples = TypeVar("NSamples")
NFeatures = TypeVar("NFeatures")

class LinearModel(Generic[NFeatures]):
    def fit(
        self, X: ndarray[float, NSamples, NFeatures], y: ndarray[float, NSamples]
    ) -> LinearModel[NFeatures]: ...
    def predict(self, X: ndarray[float, NSamples, NFeatures]) -> ndarray[float, NSamples]: ...
