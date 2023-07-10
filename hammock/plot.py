from __future__ import annotations

from io import StringIO
from pathlib import Path
import textwrap
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)
import typing

import numpy as np
from numpy import ndarray
from plotly.graph_objects import Scatter3d
import plotly.graph_objects as go

from .cache import BytesCache, cache_with_path
from .cluster import (
    ClusterControl,
    ClusterData,
    CRColorAndSummarize,
    CRColor,
    PointColoring,
    handle_clustering,
)
from .color import distance_colors, qualitative_colors
from .embedding import EmbeddingModelName, get_embeddings, reduce_embeddings
from .util import flatmap, flatten, transpose, unstack, unzip

if TYPE_CHECKING:
    from plotly.colors import RGBStr

profile = True

output_dir = Path("output")

MarkerType: TypeAlias = Literal["circle", "square", "circle-open", "diamond", "square-open", "diamond-open"]
marker_types: Sequence[MarkerType] = typing.get_args(MarkerType)

NPoints = TypeVar("NPoints")
SmallD = TypeVar("SmallD", bound=Literal[2, 3])


class PlotData(NamedTuple, Generic[NPoints, SmallD]):
    embeddings: ndarray[float, NPoints, SmallD]
    text_fragments: Sequence[str]
    title: str
    marker_colors: Sequence[RGBStr] | RGBStr
    marker: MarkerType
    # Used when we want to toggle between different levels of clustering
    alt_colors: Optional[Sequence[PointColoring]] = None


def _cluster_marker_trace(clustering: Sequence[ClusterData[SmallD]], visible: bool, index: int) -> Scatter3d:
    """Add "passive" trace for cluster markers to the figure.
    Most will be hidden but we can use JS to toggle them on/off."""
    center_array = np.array([c.center for c in clustering])
    return go.Scatter3d(
        x=center_array[:, 0],
        y=center_array[:, 1],
        z=center_array[:, 2],
        text=[c.label for c in clustering],
        mode="markers+text",
        showlegend=False,
        hovertemplate="<extra></extra>",
        visible=visible,
        textfont=dict(
            size=16,
            family="Arial Black",
        ),
        marker=dict(
            size=4,
            color=[c.color for c in clustering],
            showscale=False,
            symbol="x",
        ),
        name=f"cluster_markers_{index}",
    )


def _marker_type_size_adjustment(marker_type: MarkerType) -> float:
    """Not all markers have the same visual impact.
    We adjust the size depending on the marker type to make them more visually similar."""

    match marker_type:
        case "circle" | "circle-open":
            return 1.0
        case "square" | "square-open":
            return 0.8
        case "diamond" | "diamond-open":
            return 0.7


def _point_traces_for_single_source(
    plot_data: PlotData[NPoints, SmallD],
    include_labels: Literal["include_labels", "exclude_labels"],
) -> Sequence[Scatter3d]:
    """A plot may have multiple sources (e.g. different books).
    This function handles all the traces that are exclusive to a single source."""
    traces: Sequence[Scatter3d] = []
    match plot_data.embeddings.shape[1]:
        case 3:
            data = plot_data.embeddings
        case 2:
            # If we have 2D embeddings, we make the third dimension a "chronological" dimension
            # reflecting the original text order.
            data = np.insert(
                plot_data.embeddings, 0, np.arange(plot_data.embeddings.shape[0], dtype=float), axis=1
            )
        case _:
            raise ValueError(f"Unsupported number of dimensions: {plot_data.embeddings.shape[1]}")
    # Outer sequence is one element per point
    # Inner sequence is one element per alternative cluster coloring scheme
    alt_colors_as_custom_data: Sequence[Mapping[str, Sequence[RGBStr]]] = (
        [] if plot_data.alt_colors is None else [{"colors": c} for c in transpose(plot_data.alt_colors)]
    )
    traces.append(
        go.Scatter3d(
            customdata=alt_colors_as_custom_data,
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode="markers",
            showlegend=False,
            text=["<br>".join(textwrap.wrap(frag, width=120)) for frag in plot_data.text_fragments]
            if include_labels == "include_labels"
            else None,
            marker=dict(
                size=6 * _marker_type_size_adjustment(plot_data.marker),
                color=plot_data.marker_colors,
                showscale=False,
                opacity=0.4,
                symbol=plot_data.marker,
                line_width=2 if "-open" in plot_data.marker else None,
                line_color=plot_data.marker_colors,
            ),
            **cast(Any,
                dict(hovertemplate="%{text}<extra></extra>")
                if include_labels == "include_labels"
                else dict(hoverinfo="none")
            ),
        )
    )
    # If we have linear embeddings, add a little marker highlighting the first text fragment
    if plot_data.embeddings.shape[1] == 2:
        traces.append(
            go.Scatter3d(
                x=[data[0, 0]],
                y=[data[0, 1]],
                z=[data[0, 2]],
                mode="markers",
                showlegend=False,
                text=["Start"],
                hovertemplate="%{text}<extra></extra>",
                marker=dict(symbol="circle-open", size=10, color="yellow", line=dict(color="yellow", width=2)),
            )
        )
    return traces


# HTML in `bytes` form
HTMLBytes = NewType("HTMLBytes", bytes)
output_dir = Path("output")
# pyright just types this as `dict[str, str]` if we declare it inline
bytes_html: BytesCache = {"format": "bytes", "ext": "html"}


@cache_with_path(output_dir, bytes_html)
def _plot(
    plot_datas: Sequence[PlotData[NPoints, SmallD]],
    clusterss: Sequence[Sequence[ClusterData[SmallD]]],
    include_script: Literal["include_script", "exclude_script"] = "include_script",
    include_labels: Literal["include_labels", "exclude_labels"] = "include_labels",
) -> HTMLBytes:
    fig = go.Figure()

    titles = [pt.title for pt in plot_datas]
    for trace in flatmap(lambda p: _point_traces_for_single_source(p, include_labels=include_labels), plot_datas):
        fig.add_trace(trace)
    for i, clustering in enumerate(clusterss):
        fig.add_trace(_cluster_marker_trace(clustering, visible=(i == 0), index=i))

    background_color = "rgb(168, 168, 192)"

    fig.add_annotation(
        text=(
            (
                "Use the left and right keys on the keyboard to step through fragments.<br>"
                if include_labels == "include_labels"
                else ""
            )
            + "Use the up and down keys to change clustering level."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=0,
        showarrow=False,
    )

    invisible_axis = dict(
        showticklabels=False,
        showspikes=False,
        backgroundcolor=background_color,
        gridcolor=background_color,
        zerolinecolor=background_color,
    )

    # Make the plot a a vast, formless void
    fig.update_layout(
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            xaxis=invisible_axis,
            yaxis=invisible_axis,
            zaxis=invisible_axis,
        ),
        title="<br>".join(titles),
    )

    with StringIO() as s:
        match include_script:
            case "include_script":
                with open("templates/plotly.js", encoding="utf-8") as f:
                    fig.write_html(s, include_plotlyjs="cdn", post_script=f.read())
            case "exclude_script":
                fig.write_html(s, include_plotlyjs="cdn")
        return HTMLBytes(s.getvalue().encode("utf-8"))


class Source(NamedTuple):
    title: str
    text_fragments: list[str]


EmbeddingDim = TypeVar("EmbeddingDim")


def _mk_plot_and_cluster_data(
    unstacked_reduced_embeddings: Sequence[ndarray[float, NPoints, SmallD]],
    original_embeddings: Optional[ndarray[float, NPoints, EmbeddingDim]],
    cluster_control: ClusterControl,
    sources: Sequence[Source],
) -> tuple[Sequence[PlotData[NPoints, SmallD]], Sequence[Sequence[ClusterData[SmallD]]]]:
    titles, textss = unzip(sources)
    split_indices = np.cumsum([e.shape[0] for e in unstacked_reduced_embeddings][:-1])
    match handle_clustering(cluster_control, flatten(textss), np.vstack(unstacked_reduced_embeddings)):
        case None:
            return [
                PlotData(*t)
                for t in zip(
                    unstacked_reduced_embeddings,
                    textss,
                    titles,
                    # Either qualitatively by source or
                    # by distance in the original (i.e non-reduced) embedding dimension
                    qualitative_colors(len(unstacked_reduced_embeddings))
                    if original_embeddings is None
                    else unstack(distance_colors(original_embeddings), split_indices.tolist()),
                    marker_types[: len(unstacked_reduced_embeddings)],
                    strict=True,
                )
            ], []
        case [CRColorAndSummarize(), *_] as result:
            return (
                [
                    PlotData(*t)
                    for t in zip(
                        unstacked_reduced_embeddings,
                        textss,
                        titles,
                        unstack(result[0].colors, split_indices.tolist()),
                        marker_types[: len(unstacked_reduced_embeddings)],
                        transpose([unstack(s.colors, split_indices.tolist()) for s in result]),
                        strict=True,
                    )
                ],
                [s.clusters for s in result],
            )
        case [CRColor(), *_] as colors:
            return (
                [
                    PlotData(*t)
                    for t in zip(
                        unstacked_reduced_embeddings,
                        textss,
                        titles,
                        unstack(colors[0].colors, split_indices.tolist()),
                        marker_types[: len(unstacked_reduced_embeddings)],
                        transpose([unstack(c.colors, split_indices.tolist()) for c in colors]),
                        strict=True,
                    )
                ],
                [],
            )
        case []:
            raise RuntimeError("Unexpected empty list in plot_multiple")
        case _:
            raise AssertionError("Pyright can't tell this is already exhaustive")


def plot_single(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    embedding_instruction: str,
    dimensions: Literal[2, 3],
    cluster_control: ClusterControl,
    source: Source,
    include_labels: Literal["include_labels", "exclude_labels"] = "include_labels",
) -> Path:
    embeddings = get_embeddings(embedding_model_name, embedding_instruction, source.text_fragments)
    reduced_embeddings = reduce_embeddings(dimensions, n_neighbors=n_neighbors, embeddings=embeddings)
    plot_data, cluster_data = _mk_plot_and_cluster_data(
        [reduced_embeddings], embeddings, cluster_control, [source]
    )
    return _plot(
        plot_data, cluster_data, include_script="include_script", include_labels=include_labels
    ).cache_path


n_neighbors = 10


def plot_multiple(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    embedding_instruction: str,
    dimensions: Literal[2, 3],
    cluster_control: ClusterControl,
    sources: Sequence[Source],
    include_labels: Literal["include_labels", "exclude_labels"] = "include_labels",
) -> Path:
    embeddings = [
        get_embeddings(embedding_model_name, embedding_instruction, source.text_fragments) for source in sources
    ]
    stacked_embeddings = reduce_embeddings(dimensions, n_neighbors=n_neighbors, embeddings=np.vstack(embeddings))
    split_indices = np.cumsum([e.shape[0] for e in embeddings][:-1])
    unstacked_embeddings = np.vsplit(stacked_embeddings, split_indices)
    plot_data, cluster_data = _mk_plot_and_cluster_data(unstacked_embeddings, None, cluster_control, sources)
    return _plot(
        plot_data, cluster_data, include_script="include_script", include_labels=include_labels
    ).cache_path
