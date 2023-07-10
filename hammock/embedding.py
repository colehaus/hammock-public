from __future__ import annotations

import functools
from pathlib import Path
from typing import Generic, Literal, NamedTuple, Protocol, TypeVar, cast
import warnings

from InstructorEmbedding import INSTRUCTOR
from numba.core.errors import NumbaDeprecationWarning
from numpy import ndarray
from scipy.spatial.distance import pdist, squareform
from sentence_transformers import SentenceTransformer

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    from umap import UMAP

from .cache import cache
from .util import time_segment

profile = True

embeddings_dir = Path("cache/embeddings")

NTexts = TypeVar("NTexts")
EmbeddingDim = TypeVar("EmbeddingDim")
SmallD = TypeVar("SmallD", bound=Literal[2, 3])


# Just some convenience protocols to make our `_embedding_model` type signature a little more readable
class Embedder(Protocol[EmbeddingDim]):
    def __call__(self, text_fragments: list[str]) -> ndarray[float, NTexts, EmbeddingDim]:
        ...


class MkEmbedder(Protocol[EmbeddingDim]):
    def __call__(self, embedding_instruction: str) -> Embedder[EmbeddingDim]:
        ...


@functools.cache
def _embedding_model(model_desc: EmbeddingModelName[EmbeddingDim]) -> MkEmbedder[EmbeddingDim]:
    """The type signature is a little clumsy here but we want to ensure our `@functools.cache` shares
    the same model instance across all calls with a given `model_name`."""
    if model_desc.name.startswith("hkunlp/instructor"):
        instructor: INSTRUCTOR[EmbeddingDim] = INSTRUCTOR(model_desc.name)
        return lambda embedding_instruction: lambda text_fragments: instructor.encode(
            [[embedding_instruction, t] for t in text_fragments],
            batch_size=2,
            show_progress_bar=True,
        )
    else:
        st: SentenceTransformer[EmbeddingDim] = SentenceTransformer(model_desc.name)
        return lambda embedding_instruction: lambda text_fragments: st.encode(
            text_fragments, batch_size=2, show_progress_bar=True
        )


@cache(embeddings_dir)
def get_embeddings(
    model_name: EmbeddingModelName[EmbeddingDim],
    embedding_instruction: str,
    text_fragments: list[str],
) -> ndarray[float, NTexts, EmbeddingDim]:
    with time_segment("embedding", active=profile):
        # Have to cast to tie EmbeddingDim@get_embeddings to EmbeddingDim@_embedding_model
        # In theory, we could attach the `EmbeddingDim` to `EmbeddingModelName` as a phantom type,
        # but that seems like it's more work than it's worth ATM.
        return cast(
            "ndarray[float, NTexts, EmbeddingDim]",
            _embedding_model(model_name)(embedding_instruction)(text_fragments),
        )


# Can't load these because they're too big for my laptop
# "hkunlp/instructor-xl"
# "sentence-t5-xxl"
# Doesn't seem that great?
# i.e. Pareto-dominated by "instructor" line which seem get better performance at smaller size.
# "sentence-t5-xl"


class EmbeddingModelName(NamedTuple, Generic[EmbeddingDim]):
    # hkunlp/instructors are clearly better but notably slower. The other two can be nice for quick testing.
    name: Literal["hkunlp/instructor-large", "hkunlp/instructor-base", "all-mpnet-base-v2", "all-MiniLM-L6-v2"]


instructor_large = EmbeddingModelName[Literal["instructor-large-dim"]]("hkunlp/instructor-large")
instructor_base = EmbeddingModelName[Literal["instructor-base-dim"]]("hkunlp/instructor-base")
all_MP_net_base = EmbeddingModelName[Literal["mpnet-base-dim"]]("all-mpnet-base-v2")
all_mini_LM = EmbeddingModelName[Literal["minilm-dim"]]("all-MiniLM-L6-v2")


@cache(embeddings_dir)
def reduce_embeddings(
    dimensions: Literal[2, 3], n_neighbors: int, embeddings: ndarray[float, NTexts, EmbeddingDim]
) -> ndarray[float, NTexts, SmallD]:
    """Use UMAP to reduce the dimensionality of the embeddings to 2 or 3 dimensions.
    `n_neighbors` seems to be the most important parameter for our use case:
    https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
    """
    with time_segment(f"UMAP {dimensions}D", active=profile):
        # segfaults if we don't precompute for values larger than 4096 (on some data)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="using precomputed metric; inverse_transform will be unavailable"
            )
            return UMAP(
                n_neighbors=n_neighbors,
                min_dist=0,
                n_components=cast(SmallD, dimensions),
                metric="precomputed",
                verbose=True,
            ).fit_transform(squareform(pdist(embeddings, metric="euclidean")))
