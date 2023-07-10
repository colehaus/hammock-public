from itertools import combinations
from pathlib import Path
from typing import Callable, Generic, Literal, NamedTuple, Sequence, TypeVar

from nltk.tokenize import word_tokenize
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy import ndarray
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ..core import para_tokenize


def get_core(book_path: Path) -> tuple[str, str]:
    """Books generally have a preamble and a postamble that we're not really interested in.
    We want to automatically prune that content.
    We do that heuristically by looking at the number of words per paragraph since
    the preamble and postamble tend to have shorter paragraphs.
    We pick a breakpoint in the distribution of paragraph lengths and
    then find the longest contiguous section of paragraphs that are above that breakpoint."""
    with open(book_path, encoding="utf-8") as f:
        text = f.read()
    paras = para_tokenize("double", text)
    word_counts = np.array([len(word_tokenize(para)) for para in paras])
    percentiles = np.linspace(0, 100, 101)
    percentile_values = np.percentile(word_counts, percentiles)
    # Sometimes the very ends of the percentile graphs get erroneously registered as break points.
    # So we clip them.
    percentile = find_breakpoint(percentiles[2:-2], percentile_values[2:-2])
    percentile_margin = 1
    threshold = np.percentile(word_counts, percentile + percentile_margin)
    start, end = longest_contiguous_above_threshold(
        word_counts, threshold, supermajority=0.4, window_size=20, end_margin=0.05
    )
    return paras[start], paras[end]
    # print(book_path.stem, start, end)
    # plot_diagnostic(
    #     word_counts,
    #     paras,
    #     AnalysisResults(
    #         percentiles=percentiles,
    #         percentile_values=percentile_values,
    #         chosen_percentile=percentile + percentile_margin,
    #         start=start,
    #         end=end,
    #         threshold=threshold,
    #     ),
    #     Path(f"tmp-out/{book_path.stem}.html"),
    # )


NPercentiles = TypeVar("NPercentiles")


class AnalysisResults(NamedTuple, Generic[NPercentiles]):
    percentiles: ndarray[float, NPercentiles]
    percentile_values: ndarray[float, NPercentiles]
    chosen_percentile: float
    start: int
    end: int
    threshold: float


def plot_diagnostic(
    word_counts: Sequence[int], paras: Sequence[str], results: AnalysisResults[NPercentiles], out_path: Path
):
    """Visualizes multiple pieces of data to diagnose our heuristics.
    - Plot showing the word counts by paragraph, in order
    - Plot showing the percentile values by percentile
    - Vertical line on percentile plot showing the percentile chosen by breakpoint detection
    - Horizontal lines on word plot showing the correspoding inferred threshold of words per paragraph
    - Vertical lines on word plot showing the start and end of the longest contiguous region above the threshold
    """
    fig = sp.make_subplots(rows=1, cols=2, subplot_titles=["Word counts", "Percentiles"])
    fig.add_trace(
        go.Scatter(x=list(range(len(word_counts))), y=word_counts, text=paras, mode="lines"),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=results.percentiles, y=results.percentile_values, mode="markers"), row=1, col=2)
    fig.update_layout(
        shapes=[
            go.layout.Shape(
                type="line",
                x0=results.chosen_percentile,
                x1=results.chosen_percentile,
                y0=0,
                y1=max(word_counts),
                xref="x2",
                yref="y2",
            ),
            go.layout.Shape(
                type="line",
                x0=results.start,
                x1=results.start,
                y0=0,
                y1=word_counts[results.start],
                xref="x1",
                yref="y1",
                line=dict(color="red"),
            ),
            go.layout.Shape(
                type="line",
                x0=results.end,
                x1=results.end,
                y0=0,
                y1=word_counts[results.end],
                xref="x1",
                yref="y1",
                line=dict(color="red"),
            ),
            go.layout.Shape(
                type="line",
                x0=0,
                x1=len(word_counts),
                y0=results.threshold,
                y1=results.threshold,
                xref="x1",
                yref="y1",
            ),
        ]
    )
    fig.write_html(file=out_path)


A = TypeVar("A")
X = TypeVar("X")
WinSize = TypeVar("WinSize", bound=int)


def in_window_satisfying_predicate(
    values: ndarray[A, X], window_size: WinSize, predicate: Callable[[ndarray[A, X, WinSize]], ndarray[bool, X]]
) -> ndarray[bool, X]:
    """Returns a boolean array indicating whether each element is in any rolling window
    where that window satisfies the predicate."""

    windows = sliding_window_view(values, window_size)
    bools = predicate(windows)
    result = np.zeros(len(values), dtype=bool)
    for i in range(window_size):
        result[i : -(window_size - i - 1) if window_size - i - 1 > 0 else None] |= bools
    return result


NParas = TypeVar("NParas")


def find_breakpoint(x: ndarray[float, NParas], y: ndarray[float, NParas]) -> float:
    """Fit two linear regressions to the data, one for each side of the breakpoint.
    Find the breakpoint that minimizes the sum of the MSEs of the two regressions."""

    def mse_for_breakpoint(i: int) -> float:
        model1 = LinearRegression[Literal[1]]().fit(x[:i].reshape((-1, 1)), y[:i])
        model2 = LinearRegression[Literal[1]]().fit(x[i:].reshape((-1, 1)), y[i:])
        mse1 = mean_squared_error(y[:i], model1.predict(x[:i].reshape((-1, 1))))
        mse2 = mean_squared_error(y[i:], model2.predict(x[i:].reshape((-1, 1))))
        return mse1 + mse2

    return x[np.argmin([mse_for_breakpoint(i) for i in range(1, len(x) - 1)]) + 1]


def segment_average_from_cumulative(segment: tuple[int, int], cumulative_counts: ndarray[int, NParas]) -> float:
    """Faster way to compute the average in a segment."""
    return (cumulative_counts[segment[1]] - cumulative_counts[segment[0] - 1]) / (segment[1] - segment[0] + 1)


def longest_contiguous_above_threshold(
    word_counts: ndarray[int, NParas],
    threshold: float,
    supermajority: float,
    window_size: int,
    end_margin: float,
) -> tuple[int, int]:
    """Roughly, Find the longest contiguous segment of paragraphs where the word count is above the threshold.
    Parameters
    ----------
    counts : Sequence[int]
        The number of words in each paragraph.
    threshold : float
        The value tending to distinguish pre- and postamble from the main body. Unit is words per paragraph.
    supermajority : float
        The fraction of paragraphs in the whole segment that must be above the threshold.
        We can't require /all/ paragraphs to be above the threshold because
        there will occasionally be short paragraphs. But we want a large fraction above the threshold.
    window_size : int
        The size of the window used to smooth the evaluation.
        We pick candidate segment endpoints based on whether their windows satisfy certain criteria.
    end_margin : float
        Our candidate segments might end up containing clusters of low word count paragraphs
        near the beginning or end of the overall content. If these clusters are too close to the ends,
        we reject the candidate segment on the theory that the low clusters are pre- or postamble
        and our candidate endpoints are actually anomalous pre- or postamble content.
    """
    in_average_above_threshold_window = in_window_satisfying_predicate(
        word_counts, window_size, lambda windows: np.mean(windows, axis=1) >= threshold
    )
    in_supermajority_above_threshold_window = in_window_satisfying_predicate(
        word_counts,
        window_size,
        lambda windows: np.mean(windows >= threshold, axis=1) >= supermajority,
    )
    # Paragraphs that are candidate endpoints of a contiguous segment satisfy three criteria:
    # 1. The average word count of their window is above the threshold.
    # 2. The supermajority of their window is above the threshold.
    # 3. Their word count is above the threshold.
    all_segments = combinations(
        [
            i
            for i in np.nonzero(in_average_above_threshold_window & in_supermajority_above_threshold_window)[0]
            if word_counts[i] >= threshold
        ],
        2,
    )

    def has_low_cluster_near_end(segment: tuple[int, int]) -> bool:
        """If the candidate segment has a cluster of short paragraphs near the beginning or end
        of the overall content, we reject it.
        This suggests that the candidate segment is actually including pre- or postamble and
        the longer paragraphs that are our segment endpoints are just anomolous pre- or postamble content."""
        if segment[1] - segment[0] < window_size:
            return True
        else:
            low_clusters = np.nonzero(
                in_window_satisfying_predicate(
                    word_counts[segment[0] : segment[1]],
                    window_size,
                    lambda windows: ~(np.mean(windows >= threshold, axis=1) >= supermajority),
                )
            )[0]
            if low_clusters.size == 0:
                return False
            else:
                earliest_low = low_clusters[0] + segment[0]
                latest_low = low_clusters[-1] + segment[0]
                return earliest_low / len(word_counts) < end_margin or latest_low / len(word_counts) > (
                    1 - end_margin
                )

    cumulative_counts = np.cumsum(word_counts)
    cumulative_threshold_counts = np.cumsum(np.where(word_counts >= threshold, 1, 0))
    valid_segments = (
        segment
        for segment in all_segments
        if segment_average_from_cumulative(segment, cumulative_counts) >= threshold and
        # Ensure that most values are above the threshold
        segment_average_from_cumulative(segment, cumulative_threshold_counts) >= supermajority
    )

    longest_segments = reversed(sorted(valid_segments, key=lambda segment: segment[1] - segment[0]))
    for s in longest_segments:
        # Special handling for this predicate because it's relatively slow
        if not has_low_cluster_near_end(s):
            return s
    raise RuntimeError("No valid segment found")


# Doesn't really work. At least with 'google/flan-t5-base' or 'google/flan-t5-large'.
# def identify_preamble(model_name: str):
#     batch_size = 1
#     prompt = (
#         "Is the focused paragraph in the input the first paragraph that's not part of a book's preamble? "
#         "i.e. not the table of contents, preface, copyright, etc. "
#         "Just answer with a percentage confidence that it is like 0.2, 0.4, 0.6, or 0.8 "
#         "The paragraphs follow: "
#     )
