# pylint: skip-file

from typing import Optional, TypeVar

import numpy as np
from numpy import ndarray

NSamples = TypeVar("NSamples")
NFeatures = TypeVar("NFeatures")

class HDBSCAN:
    def __init__(self, min_cluster_size: int = 5, min_samples: Optional[int] = None) -> None: ...
    def fit_predict(self, X: ndarray[float, NSamples, NFeatures]) -> ndarray[np.int64, NSamples]: ...
