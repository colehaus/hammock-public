# pylint: skip-file

from typing import Generic, Literal, TypeVar

from numpy import ndarray

NClusters = TypeVar("NClusters")
NSamples = TypeVar("NSamples")
NFeatures = TypeVar("NFeatures")

class KMedoids(Generic[NClusters, NFeatures]):
    def __init__(self, n_clusters: NClusters, method: Literal["pam"], init: Literal["k-medoids++"]) -> None: ...
    def fit(self, X: ndarray[float, NSamples, NFeatures]) -> KMedoids[NClusters, NFeatures]: ...
    medoid_indices_: ndarray[int, NClusters]
    cluster_centers_: ndarray[float, NClusters, NFeatures]
