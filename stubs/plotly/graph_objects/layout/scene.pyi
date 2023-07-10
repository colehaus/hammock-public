# pylint: skip-file

from typing import Optional

class Annotation:
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        text: str,
        showarrow: bool = True,
        bgcolor: Optional[str] = None,
        borderpad: Optional[float] = 1,
        visible: bool = True,
    ) -> None: ...
