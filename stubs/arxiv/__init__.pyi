# pylint: skip-file

from enum import Enum
from typing import Generator

class SortCriterion(Enum):
    Relevance = "relevance"
    LastUpdatedDate = "lastUpdatedDate"
    SubmittedDate = "submittedDate"

class Result:
    summary: str
    title: str
    pdf_url: str
    def download_pdf(self, filename: str) -> None: ...

class Search:
    def __init__(self, query: str, max_results: int, sort_by: SortCriterion) -> None: ...
    def results(self) -> Generator[Result, None, None]: ...
