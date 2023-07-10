from typing import TypeVar

from sentence_transformers import SentenceTransformer
from transformers import T5TokenizerFast

EmbeddingDim = TypeVar("EmbeddingDim")

class INSTRUCTOR(SentenceTransformer[EmbeddingDim]):
    tokenizer: T5TokenizerFast
