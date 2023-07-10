# pylint: skip-file

from io import StringIO
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, TypeVar, overload

from numpy import ndarray
from plotly.graph_objects.layout.scene import Annotation

from .layout import *

class Trace: ...

NSamples = TypeVar("NSamples")

class Scatter(Trace):
    def __init__(
        self,
        x: Sequence[float] | ndarray[float, NSamples],
        y: Sequence[float] | ndarray[float, NSamples],
        mode: Literal["markers", "lines"],
        text: Optional[Sequence[str]] = None,
    ) -> None: ...

class Scatter3d(Trace):
    def __init__(
        self,
        x: ndarray[float, NSamples] | Sequence[float],
        y: ndarray[
            float,
            NSamples,
        ]
        | Sequence[float],
        z: ndarray[float, NSamples] | Sequence[float],
        mode: Literal["markers+lines", "markers", "markers+text"],
        showlegend: bool,
        marker: dict[str, Any],
        textfont: Optional[dict[str, Any]] = None,
        visible: bool = True,
        name: Optional[str] = None,
        text: Optional[Sequence[str]] = None,
        hovertemplate: Optional[str] = None,
        hoverinfo: Optional[Literal["text", "none"]] = None,
        line: Optional[Mapping[str, Any]] = None,
        customdata: Optional[Sequence[Any]] = None,
    ) -> None: ...

class Figure:
    def __init__(self) -> None: ...
    def add_trace(self, trace: Trace, row: Optional[int] = None, col: Optional[int] = None) -> Figure: ...
    @overload
    def add_annotation(self, arg: Annotation) -> Figure: ...
    @overload
    def add_annotation(
        self,
        text: str,
        xref: Literal["paper"],
        yref: Literal["paper"],
        x: float,
        y: float,
        showarrow: Literal[False],
    ) -> Figure: ...
    def update_layout(
        self,
        scene: Optional[Mapping[str, Any]] = None,
        title: Optional[str] = None,
        hovermode: Optional[Literal["closest"]] = None,
        hoverlabel: Optional[Mapping[str, Any]] = None,
        paper_bgcolor: Optional[str] = None,
        plot_bgcolor: Optional[str] = None,
        shapes: Optional[Sequence[Shape]] = None,
    ) -> Figure: ...
    def write_html(
        self,
        file: str | Path | StringIO,
        include_plotlyjs: bool | Literal["cdn"] = True,
        div_id: Optional[str] = None,
        post_script: Optional[str] = None,
    ) -> None: ...
