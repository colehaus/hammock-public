# pylint: skip-file

from typing import Any, Literal, Mapping, Optional

class Shape:
    def __init__(
        self,
        type: Literal["line"],
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        xref: str,
        yref: str,
        line: Optional[Mapping[str, Any]] = None,
    ) -> None: ...
