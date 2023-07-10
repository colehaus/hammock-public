# pylint: skip-file

from typing import Literal, TypeVar

from numpy import ndarray

NSamples = TypeVar("NSamples")
NFeatures = TypeVar("NFeatures")

# The return type isn't quite rigth because `pdist` actually returns a condensed array but it's good enough for now
def pdist(X: ndarray[float, NSamples, NFeatures], metric: Literal["euclidean"]) -> ndarray[float, NSamples]: ...
def squareform(X: ndarray[float, NSamples]) -> ndarray[float, NSamples, NSamples]: ...
