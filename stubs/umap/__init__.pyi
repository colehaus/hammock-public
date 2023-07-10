# pylint: skip-file

from typing import Generic, TypeVar

from numpy import ndarray

NSamples = TypeVar("NSamples")
NFeatures = TypeVar("NFeatures")
NComponents = TypeVar("NComponents", bound=int)

class UMAP(Generic[NComponents]):
    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: NComponents = 2,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 0,
        verbose: bool = False,
    ) -> None: ...
    def fit_transform(self, X: ndarray[float, NSamples, NSamples]) -> ndarray[float, NSamples, NComponents]: ...
