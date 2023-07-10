from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Mapping, NewType, Sequence, TypeVar, cast

from networkx import Graph, equitable_color
import numpy as np
from numpy import ndarray
from plotly import colors
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import minmax_scale

from .util import time_segment

if TYPE_CHECKING:
    from networkx import WeightDict
    from plotly.colors import HexStr, PlotlyScales, RGBStr, Tuple1, Tuple255

profile = False
noise_cluster_id = -1

ID = TypeVar("ID")

# 2D or 3D
SmallD = TypeVar("SmallD")


def graph_from_centers(cluster_centers: Mapping[ID, ndarray[float, SmallD]], percentile: float) -> Graph[ID]:
    """Given a collection of points representing the centers of clusters,
    create an undirected graph connecting those points.
    Two cluster centers are connected if the (Euclidean) distance between them is
    less than the given percentile of all pairwise distances."""

    nodes: list[ID] = list(cluster_centers.keys())
    node_pairs: list[tuple[ID, ID]] = list(combinations(nodes, 2))
    all_distances: list[float] = [np.linalg.norm(cluster_centers[u] - cluster_centers[v]) for u, v in node_pairs]
    threshold: float = np.percentile(np.array(all_distances), percentile)
    edges: list[tuple[ID, ID, WeightDict]] = [
        (u, v, {"weight": d}) for (u, v), d in zip(node_pairs, all_distances, strict=True) if d < threshold
    ]
    graph: Graph[ID] = Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def cluster_colors_by_graph(cluster_centers: Mapping[ID, ndarray[float, SmallD]]) -> Mapping[ID, Tuple1]:
    """Given a collection of points representing the centers of clusters,
    choose colors for those clusters by coloring the nodes of the corresponding graph
    such that no two adjacent nodes have the same color."""

    graph = graph_from_centers(cluster_centers, percentile=10)
    max_degree = max(d for _, d in graph.degree())
    with time_segment("coloring", active=profile):
        cluster_label_to_color_index: Mapping[ID, int] = equitable_color(graph, max(max_degree + 1, 10))
    color_scale = colors.sample_colorscale(
        "Phase", len(set(cluster_label_to_color_index.values())), colortype="tuple"
    )
    return {k: color_scale[v] for k, v in cluster_label_to_color_index.items()}


NPoints = TypeVar("NPoints")
NDims = TypeVar("NDims")
NColors = NewType("NColors", int)
EmbeddingDim = TypeVar("EmbeddingDim")


def _colors_from_scale(
    scale_name: PlotlyScales, values: ndarray[float, NPoints]
) -> ndarray[float, NPoints, Tuple1]:
    """Plotly sequential color scales have a finite list of colors.
    We transform that into a truly continuous color scale by interpolating.
    Also, it's vectorized because this turned out to be a performance bottleneck.
    But morally, we want
    `_colors_from_scale(scale_name: PlotlyScales) -> Callable[[float], tuple[float, float, float]]`."""

    clamped = np.clip(values, 0, 1)
    colorscale = colors.convert_colors_to_same_type(getattr(colors.sequential, scale_name), "tuple")[0]
    colorscale_array = cast(ndarray[float, NColors, Tuple1], np.array(colorscale))
    # Our actual fractional 'index'
    index: ndarray[float, NPoints] = clamped * (len(colorscale) - 1)
    # Closest integral indices to pick from provided colors in scale and interpolate between
    index_low = np.floor(index)
    index_high = np.ceil(index)
    interp = index - index_low
    return np.floor(
        (
            colorscale_array[index_low.astype(int), :] * (1 - interp)[:, np.newaxis]
            + colorscale_array[index_high.astype(int), :] * interp[:, np.newaxis]
        )
    )


def _compute_distances(data: ndarray[float, NPoints, NDims]) -> ndarray[float, NPoints]:
    return np.mean(squareform(pdist(data, metric="euclidean")), axis=0)


def distance_colors(embeddings: ndarray[float, NPoints, EmbeddingDim]) -> Sequence[RGBStr]:
    """Color points using a continuous scale based on their average distance to all other points."""
    norm_distances = minmax_scale(_compute_distances(embeddings))
    return [
        colors.label_rgb(colors.convert_to_RGB_255(cast("Tuple1", c)))
        for c in _colors_from_scale("thermal", norm_distances)
    ]


def qualitative_colors(num_colors: int) -> Sequence[RGBStr]:
    def _to_rgb(x: str) -> Tuple255:
        match x:
            case x if x.startswith("rgb"):
                return colors.unlabel_rgb(cast("RGBStr", x))
            case x if x.startswith("#"):
                return colors.hex_to_rgb(cast("HexStr", x))
            case _:
                raise ValueError(f"Unknown color format: {x}")

    return [colors.label_rgb(_to_rgb(c)) for c in colors.qualitative.Plotly[:num_colors]]
