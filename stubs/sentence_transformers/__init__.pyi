# pylint: skip-file

from typing import Any, Generic, Optional, Sequence, TypeVar

from numpy import ndarray
import torch

EmbeddingDim = TypeVar("EmbeddingDim")

class SentenceTransformer(Generic[EmbeddingDim]):
    def __init__(
        self, model_name_or_path: Optional[str] = None, modules: Optional[Sequence[torch.nn.Module]] = None
    ) -> None: ...
    def encode(
        self,
        sentences: list[str] | list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> ndarray[float, Any, EmbeddingDim]: ...
