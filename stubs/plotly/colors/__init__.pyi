# pylint: skip-file

from typing import Literal, NewType, TypeAlias, overload

from .sequential import *
from .qualitative import *

PlotlyScales: TypeAlias = Literal[
    "Edge", "HSV", "Icefire", "Phase", "Rainbow", "Turbo", "Viridis", "haline", "mrybm", "thermal"
]

HexStr = NewType("HexStr", str)
# Values in string should be out 255
RGBStr = NewType("RGBStr", str)
# Values range from 0 to 1
Tuple1 = NewType("Tuple1", tuple[float, float, float])
# Values range from 0 to 255
Tuple255 = NewType("Tuple255", tuple[int, int, int])

def unlabel_rgb(colors: RGBStr) -> Tuple255: ...
def label_rgb(colors: Tuple255) -> RGBStr: ...
@overload
def convert_colors_to_same_type(
    colors: PlotlyScales, colortype: Literal["tuple"]
) -> tuple[list[Tuple1], None]: ...
@overload
def convert_colors_to_same_type(colors: PlotlyScales, colortype: Literal["rgb"]) -> tuple[list[RGBStr], None]: ...
def convert_to_RGB_255(colors: Tuple1) -> Tuple255: ...
def hex_to_rgb(hex: HexStr) -> Tuple255: ...
@overload
def sample_colorscale(
    colorscale: PlotlyScales, samplepoints: int, colortype: Literal["tuple"]
) -> list[Tuple1]: ...
@overload
def sample_colorscale(
    colorscale: PlotlyScales, samplepoints: int, colortype: Literal["rgb"] = "rgb"
) -> list[RGBStr]: ...
def find_intermediate_color(lowcolor: Tuple1, highcolor: Tuple1, intermed: float) -> Tuple1: ...
