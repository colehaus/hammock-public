from __future__ import annotations

from collections import defaultdict
import functools
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    Mapping,
    NamedTuple,
    NewType,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)

from hdbscan import HDBSCAN
from sklearn_extra.cluster import KMedoids
import numpy as np
from numpy import ndarray
from plotly import colors
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding

from .cache import cache, json_cache
from .color import cluster_colors_by_graph
from .util import flatmap, time_segment

if TYPE_CHECKING:
    from plotly.colors import RGBStr, Tuple1, Tuple255

profile = False

A = TypeVar("A")

ClusterID = NewType("ClusterID", np.int64)

NPoints = TypeVar("NPoints")
SmallD = TypeVar("SmallD", bound=Literal[2, 3])
NClusterPoints = TypeVar("NClusterPoints")


class ClusterResult(NamedTuple, Generic[A, NPoints, NClusterPoints, SmallD]):
    cluster_by_point: ndarray[ClusterID, NPoints]
    # `NClusterPoints` is slightly sketchy because it's not the case that
    # each cluster has the same number of points. But this is at least better than `Any`.
    points_by_cluster: Mapping[ClusterID, tuple[list[A], ndarray[float, NClusterPoints, SmallD]]]


def _cluster(
    min_cluster_size: int, items: list[A], embeddings: ndarray[float, NPoints, SmallD]
) -> ClusterResult[A, NPoints, NClusterPoints, SmallD]:
    """Each item `A` in `items` should correspond to a row in `embeddings`.
    Returns a mapping from cluster labels to a tuple of the items in that cluster and their embeddings."""

    cluster_ids = cast(
        ndarray[ClusterID, NPoints],
        (HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1).fit_predict(embeddings)),
    )
    items_by_cluster: dict[ClusterID, list[A]] = defaultdict(list)
    for label, item in zip(cluster_ids, items, strict=True):
        items_by_cluster[label].append(item)
    return ClusterResult(
        cluster_ids,
        {
            label: (
                items_for_cluster,
                embeddings[cluster_ids == label, :],
            )
            for label, items_for_cluster in items_by_cluster.items()
        },
    )


NMedoids = TypeVar("NMedoids", bound=int)


def _search_for_threshold(
    first_guess: int,
    lower_bound: int,
    upper_bound: int,
    fn: Callable[[int], A],
    pred: Callable[[A], bool],
) -> A:
    """Binary search for the threshold where `pred(fn(i))` is false and `pred(fn(i+1))` is true.
    If `pred` can't be satisfied, returns largest value seen."""
    current_guess = first_guess
    smallest_saturater: tuple[int, A] | None = None
    largest_non_saturater: A | None = None
    while lower_bound <= upper_bound:
        a = fn(current_guess)
        if pred(a):
            # `lower_bound - 1` must not have saturated so the current value is a solution
            if lower_bound == current_guess:
                return a
            else:
                smallest_saturater = current_guess, a
                upper_bound = current_guess - 1
                current_guess = (lower_bound + upper_bound) // 2
        else:
            if smallest_saturater is not None and smallest_saturater[0] == current_guess + 1:
                return smallest_saturater[1]
            else:
                largest_non_saturater = a
                lower_bound = current_guess + 1
                current_guess = (lower_bound + upper_bound) // 2
    match largest_non_saturater:
        case None:
            raise RuntimeError("Should be impossible. In `_search_for_threshold`")
        case a:
            return a


def choose_reps_within_length(
    model_name: str,
    prompt: str,
    max_length: int,
    texts: Sequence[str],
    embeddings: ndarray[float, NClusterPoints, SmallD],
) -> tuple[Sequence[str], ndarray[float, NMedoids, SmallD]]:
    """The summarization model has a maximum or effective maximum input length.
    We want to choose good representatives from the cluster that total up to that length.
    We do that via k-medoids clustering.
    (k-medoids is like k-means, but the cluster centers are actual data points which is essential here.)"""

    def k_medoids(num_reps: int) -> KMedoids[NMedoids, SmallD]:
        km: KMedoids[NMedoids, SmallD] = KMedoids(
            n_clusters=cast(NMedoids, min(len(texts), num_reps)), method="pam", init="k-medoids++"
        )
        return km.fit(embeddings)

    def len_of_reps(kmeds: KMedoids[NMedoids, SmallD]) -> int:
        reps = [texts[idx] for idx in kmeds.medoid_indices_]
        ret = len(batch_encode_clusters(model_name, prompt, [reps], max_length=max_length)["input_ids"][0])
        return ret

    with time_segment(f"Choosing reps for sequence of length {len(texts)}", active=profile):
        avg_fragment_length = sum(len(text) for text in texts) / len(texts)
        first_guess = int(max_length / avg_fragment_length) - 1
        meds = _search_for_threshold(
            first_guess,
            lower_bound=2,
            upper_bound=int(max_length / avg_fragment_length * 10),
            fn=k_medoids,
            pred=lambda x: len_of_reps(x) == max_length,
        )
        return [texts[idx] for idx in meds.medoid_indices_], meds.cluster_centers_


def _cluster_centers(
    embeddings: ndarray[float, NPoints, SmallD], cluster_ids: ndarray[ClusterID, NPoints]
) -> Mapping[ClusterID, ndarray[float, SmallD]]:
    """Each row in `embeddings` should correspond to a cluster label in `cluster_labels`."""

    cluster_label_array = cluster_ids
    return {
        cluster_id: np.mean(embeddings[cluster_label_array == cluster_id, :], axis=0)
        for cluster_id in set(cluster_ids)
    }


noise_cluster_id = np.int64(-1)

summary_dir = Path("cache/summary")


def batch_encode_clusters(
    model_name: str, prompt: str, clusters: Sequence[Sequence[str]], max_length: int
) -> BatchEncoding:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        [f"{prompt}\n" + "\n".join(cluster) for cluster in clusters],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


@functools.cache
def _summary_model(model_name: SummaryModelName) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    return (
        AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True),
        AutoTokenizer.from_pretrained(model_name),
    )


def _capwords_preserve_case(s: str):
    """Capitalize the first letter of each word in a string.
    (`.capwords()` will lowercase the rest of the word which doesn't work for e.g. 'AI' or 'NP'.)"""
    return " ".join(word[0].upper() + word[1:] for word in s.split())


@cache(summary_dir, cache_type=json_cache)
def _summarize_clusters(
    model_name: SummaryModelName,
    max_length: int,
    prompt: str,
    clusters: Mapping[ClusterID, tuple[Sequence[str], ndarray[float, NClusterPoints, SmallD]]],
) -> Mapping[ClusterID, str]:
    """Passes text fragments from cluster to summarization model and returns summaries.
    In `clusters`, we expect each row of the embeddings array to correspend to one text fragment `str` in the list.
    We need the embeddings so that we can do k-medoids clustering and choose good representatives for each cluster.
    (We can't just summarize /every/ text fragment in the cluster because
    the summarization model has a maximum input length.)
    """
    print(f"Num clusters: {len(clusters.keys())}")

    model, tokenizer = _summary_model(model_name)
    tokenized_input = batch_encode_clusters(
        model_name,
        prompt,
        [
            choose_reps_within_length(model_name, prompt, max_length, *v)[0]
            for k, v in clusters.items()
            if k != noise_cluster_id
        ],
        max_length=max_length,
    )
    batch_size = 1
    batch_count = 0

    def generate(input_ids: torch.Tensor) -> Sequence[str]:
        nonlocal batch_count
        batch_count += 1
        # print(f"Batch: {batch_count}")
        # for input_ in tokenizer.batch_decode(input_ids, skip_special_tokens=True):
        #     print(input_)
        # print("=======================================")

        with time_segment(f"summary batch. Size of {batch_size}", active=profile):
            batch_summary_ids = model.generate(
                input_ids,
                min_length=2,
                max_length=20,
                length_penalty=0.8,
                repetition_penalty=3.0,
            )
        batch_summary_texts = [
            _capwords_preserve_case(t.strip())
            for t in tokenizer.batch_decode(batch_summary_ids, skip_special_tokens=True)
        ]
        print(batch_summary_texts)
        return batch_summary_texts

    return dict(
        zip(
            list(clusters.keys()),
            flatmap(generate, torch.split(tokenized_input["input_ids"], batch_size)),
            strict=True,
        )
    )


# These two summarization models are much worse than t2t:
# "facebook/bart-large-cnn"
# "sshleifer/distilbart-cnn-12-6"
# Too big to run consistently
# "google/flan-t5-xl"

SummaryModelName: TypeAlias = Literal["google/flan-t5-large", "google/flan-t5-base"]

# Callers may request different clustering functionality.
# They can describe the requested functionality via the `ClusterControl` types.
# We then "interpret" the clustering specification into different `ClusterResult` types via `handle_clustering`.


class CCColorAndSummarize(NamedTuple):
    model_name: SummaryModelName
    summary_prompt: str
    min_cluster_sizes: Sequence[int]


class CCNeither(NamedTuple):
    ...


class CCColor(NamedTuple):
    min_cluster_sizes: Sequence[int]


ClusterControl: TypeAlias = CCColorAndSummarize | CCNeither | CCColor

# Color for each point
PointColoring = NewType("PointColoring", "Sequence[RGBStr]")


class CRColor(NamedTuple):
    # Color for each point
    colors: PointColoring


class ClusterData(NamedTuple, Generic[SmallD]):
    center: ndarray[float, SmallD]
    label: str
    color: RGBStr


class CRColorAndSummarize(NamedTuple, Generic[SmallD]):
    clusters: Sequence[ClusterData[SmallD]]
    # Color for each point according to its cluster
    # (so the color info here isn't duplicative with the info in `clusters` which is only one per cluster)
    colors: PointColoring


ID = TypeVar("ID")


def compute_cluster_info(
    min_cluster_size: int,
    texts: list[str],
    reduced_embeddings: ndarray[float, NPoints, SmallD],
) -> tuple[
    ndarray[ClusterID, NPoints],
    Mapping[ClusterID, tuple[Sequence[str], ndarray[float, Any, SmallD]]],
    Mapping[ClusterID, ndarray[float, SmallD]],
    Mapping[ClusterID, Tuple1],
]:
    clusters: ClusterResult[str, NPoints, Any, SmallD] = _cluster(min_cluster_size, texts, reduced_embeddings)
    centers = {
        k: v
        for k, v in _cluster_centers(reduced_embeddings, clusters.cluster_by_point).items()
        if k != noise_cluster_id
    }
    colors = cluster_colors_by_graph(centers)
    points_by_cluster = {k: v for k, v in clusters.points_by_cluster.items() if k != noise_cluster_id}
    return (
        clusters.cluster_by_point,
        points_by_cluster,
        centers,
        colors,
    )


def handle_clustering(
    cluster_control: ClusterControl, texts: list[str], reduced_embeddings: ndarray[float, NPoints, SmallD]
) -> Sequence[CRColorAndSummarize[SmallD]] | Sequence[CRColor] | None:
    """One cluster result per `min_cluster_size`.
    This allows us to view the plot at different clustering granularities."""

    def color_by_point_from_cluster(
        cluster_by_point: ndarray[ClusterID, NPoints], cluster_to_color: Mapping[ClusterID, Tuple1]
    ) -> PointColoring:
        return PointColoring(
            [
                colors.label_rgb(
                    cast("Tuple255", (255, 255, 255))
                    if cluster_id == noise_cluster_id
                    else colors.convert_to_RGB_255(cluster_to_color[cluster_id])
                )
                for cluster_id in cluster_by_point
            ]
        )

    match cluster_control:
        case CCColor(min_cluster_size):

            def _c_inner(min_cluster_size: int):
                cluster_by_point, _, _, cluster_to_color = compute_cluster_info(
                    min_cluster_size, texts, reduced_embeddings
                )
                return CRColor(colors=color_by_point_from_cluster(cluster_by_point, cluster_to_color))

            return [_c_inner(mcs) for mcs in min_cluster_size]
        case CCColorAndSummarize(summary_model_name, summary_prompt, min_cluster_size):

            def _cs_inner(min_cluster_size: int):
                cluster_by_point, clusters, cluster_to_center, cluster_to_color = compute_cluster_info(
                    min_cluster_size, texts, reduced_embeddings
                )
                summaries = _summarize_clusters(
                    summary_model_name,
                    max_length=2048,
                    prompt=summary_prompt,
                    clusters=clusters,
                )
                assert cluster_to_center.keys() == summaries.keys(), (cluster_to_center.keys(), summaries.keys())
                assert cluster_to_center.keys() == cluster_to_color.keys(), (
                    cluster_to_center.keys(),
                    cluster_to_color.keys(),
                )
                return CRColorAndSummarize(
                    [
                        ClusterData(
                            cluster_to_center[k],
                            summaries[k],
                            colors.label_rgb(colors.convert_to_RGB_255(cluster_to_color[k])),
                        )
                        for k in cluster_to_center.keys()
                    ],
                    color_by_point_from_cluster(cluster_by_point, cluster_to_color),
                )

            return [_cs_inner(mcs) for mcs in min_cluster_size]
        case CCNeither():
            return None
