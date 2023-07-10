# pylint: skip-file

from __future__ import annotations

from typing import Literal, Optional

from torch import Tensor

class T5TokenizerFast:
    def tokenize(self, text: str) -> list[str]: ...

class BatchEncoding:
    def __getitem__(self, key: str) -> Tensor: ...

class AutoTokenizer:
    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, local_files_only: bool = False) -> AutoTokenizer: ...
    def __call__(
        self,
        text: str | list[str],
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        return_tensors: Optional[Literal["pt"]] = None,
        padding: Optional[Literal["max_length", True]] = None,
    ) -> BatchEncoding: ...
    def decode(self, token_ids: Tensor, skip_special_tokens: bool = False) -> str: ...
    def batch_decode(self, token_ids: Tensor, skip_special_tokens: bool = False) -> list[str]: ...

class PretrainedConfig:
    hidden_size: int

class AutoModelForSeq2SeqLM:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        low_cpu_mem_usage: bool = False,
        local_files_only: bool = False,
    ) -> AutoModelForSeq2SeqLM: ...
    def generate(
        self,
        input_ids: Tensor,
        min_length: int = 0,
        max_length: int = 20,
        length_penalty: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0,
    ) -> Tensor: ...
    config: PretrainedConfig
