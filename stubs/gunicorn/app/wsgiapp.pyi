# pylint: skip-file

from typing import Any

from gunicorn.config import Config

class WSGIApplication:
    def init(self, parser: Any, opts: Any, args: list[Any]) -> None: ...
    def run(self) -> None: ...
    cfg: Config
