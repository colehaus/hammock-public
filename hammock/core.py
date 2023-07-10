from __future__ import annotations

from pathlib import Path
import re
from typing import (
    Callable,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)

from gutenbergpy.gutenbergcache import GutenbergCache
from gutenbergpy.textget import get_text_by_id, strip_headers
import nltk
from nltk.tokenize import sent_tokenize
import wikipedia

from .cache import cache
from .cluster import ClusterControl, CCNeither
from .embedding import EmbeddingModelName
from .plot import Source, plot_single, plot_multiple
from .util import Either, Failure, Success, flatmap, thunkify, time_segment, unzip

nltk.download("punkt", quiet=True)

profile = True

EmbeddingDim = TypeVar("EmbeddingDim")


class TextUnit(NamedTuple):
    unit: Literal["paragraph", "sentence"]
    # Even sentences get a `para_newline_type` because
    # `sent_tokenize` sometimes considers double newlines to be part of a single paragraph.
    # So we manually split on paragraphs first.
    para_newline_type: NewlineType


NewlineType: TypeAlias = Literal["double", "single"]


def para_tokenize(newline_type: NewlineType, text: str) -> list[str]:
    """Split a text block into paragraphs.
    Some sources use double newlines to separate paragraphs and some use single newlines."""
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n" if newline_type == "double" else "\n")
        if paragraph.strip() != ""
    ]


def tokenize(
    text_unit: TextUnit,
    always_bad_sentence_predicates: list[Callable[[str], bool]],
    standalone_bad_sentence_predicates: list[Callable[[str], bool]],
    bad_para_predicates: list[Callable[[str], bool]],
    text: str,
) -> list[str]:
    """Some sentences we always want to filter out (e.g. chapter declarations)
    while some are okay in the context of a paragraph but not as standalone sentences
    (e.g. really short sentences).
    We control this filtering via the predicate arguments here."""

    # We paragraph tokenize first because
    # `sent_tokenize` sometimes considers double newlines to be part of a single sentence.
    paras = para_tokenize(text_unit.para_newline_type, text)
    match text_unit.unit:
        case "sentence":
            return [
                sent
                for sent in flatmap(sent_tokenize, paras)
                if not any(p(sent) for p in always_bad_sentence_predicates + standalone_bad_sentence_predicates)
            ]
        case "paragraph":

            def sentence_preds_for_para(para: list[str]) -> list[Callable[[str], bool]]:
                match len(para):
                    case 0 | 1:
                        return always_bad_sentence_predicates + standalone_bad_sentence_predicates
                    case _:
                        return always_bad_sentence_predicates

            sentences_for_paras: list[list[str]] = [sent_tokenize(p) for p in paras]
            filtered_sentences_for_paras: list[list[str]] = [
                [
                    sent
                    for sent in sentences_for_para
                    if not any(p(sent) for p in sentence_preds_for_para(sentences_for_para))
                ]
                for para, sentences_for_para, in zip(paras, sentences_for_paras, strict=True)
                if not any(p(para) for p in bad_para_predicates)
            ]
            return [" ".join(sentences) for sentences in filtered_sentences_for_paras if len(sentences) > 0]


def clip_string(s: str, start_anchor: str, end_anchor: str) -> Optional[str]:
    """Some texts have preamble and postamble that we want to ignore.
    We do this by clipping the text to the specified anchors."""
    match re.compile(f"({re.escape(start_anchor)}.*?{re.escape(end_anchor)})", re.DOTALL).search(s):
        case None:
            return None
        case res:
            return res.group(1)


# Freeform


def plot_freeform(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    embedding_instruction: str,
    texts: list[str],
    cluster_control: ClusterControl = CCNeither(),
    include_labels: Literal["include_labels", "exclude_labels"] = "include_labels",
) -> Path:
    return plot_single(
        embedding_model_name,
        embedding_instruction,
        dimensions=3,
        cluster_control=cluster_control,
        source=Source("Freeform", texts),
        include_labels=include_labels,
    )


# Project Gutenberg


def gutenberg_embedding_instruction(text_unit: TextUnit):
    return f"Represent the Fiction {text_unit.unit} for clustering by theme: "


class GutenbergArgs(NamedTuple):
    title: str
    start_anchor: str
    end_anchor: str


GutenbergID = NewType("GutenbergID", int)


def get_gutenberg_text(gutenberg_id: GutenbergID) -> str:
    return strip_headers(get_text_by_id(gutenberg_id)).decode("utf-8")


gutenberg_cache = thunkify(GutenbergCache.get_cache)


def fetch_gutenberg(title: str, text_unit: TextUnit, start_anchor: str, end_anchor: str) -> Either[str, list[str]]:
    ids = cast(list[GutenbergID], gutenberg_cache().query(titles=[title]))
    text = None
    gutenberg_id = None
    for i in ids:
        try:
            text = get_gutenberg_text(i)
            gutenberg_id = i
            break
        except Exception as e:
            print(
                f"Failed while fetching Project Guterberg book {title} at ID {i}:\n {e}\n"
                f"Continuing with IDs from {ids}"
            )
            continue
    if text is None or gutenberg_id is None:
        return Failure(f"Could not find {title}")
    clipped = clip_string(text, start_anchor, end_anchor)
    if clipped is None:
        return Failure(f"Could not find {start_anchor} or/and {end_anchor} in {title}")
    always_bad_sentence_predicates: list[Callable[[str], bool]] = [
        lambda sent: "chapter" in sent.lower(),
        lambda sent: "book" in sent.lower(),
    ]
    standalone_bad_sentence_predicates: list[Callable[[str], bool]] = [
        lambda sent: len(sent.split()) < 3,
        lambda sent: not any(char.islower() for char in sent),
    ]
    return Success(
        tokenize(
            text_unit,
            always_bad_sentence_predicates=always_bad_sentence_predicates,
            standalone_bad_sentence_predicates=standalone_bad_sentence_predicates,
            bad_para_predicates=[],
            text=clipped,
        )
    )


def plot_single_gutenberg(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    text_unit: TextUnit,
    cluster_control: ClusterControl,
    title: str,
    start_anchor: str,
    end_anchor: str,
) -> Either[str, Path]:
    match fetch_gutenberg(title, text_unit, start_anchor, end_anchor):
        case Failure() as f:
            return f
        case Success(texts):
            return Success(
                plot_single(
                    embedding_model_name,
                    gutenberg_embedding_instruction(text_unit),
                    dimensions=2,
                    cluster_control=cluster_control,
                    source=Source(title, texts),
                    include_labels="include_labels",
                )
            )


def plot_multiple_gutenberg(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    text_unit: TextUnit,
    cluster_control: ClusterControl,
    gutenberg_argss: Sequence[GutenbergArgs],
) -> Path:
    titles, _, _ = unzip(gutenberg_argss)
    textss = [
        x.value
        for x in [fetch_gutenberg(text_unit=text_unit, **args._asdict()) for args in gutenberg_argss]
        if not isinstance(x, Failure)
    ]
    return plot_multiple(
        embedding_model_name,
        gutenberg_embedding_instruction(text_unit),
        dimensions=2,
        cluster_control=cluster_control,
        sources=[Source(title, texts) for title, texts in zip(titles, textss, strict=True)],
        include_labels="include_labels",
    )


# Wikipedia


def wiki_embedding_instruction(text_unit: TextUnit):
    return f"Represent the Wikipedia {text_unit.unit} for clustering by topic: "


def wiki_summary_prompt(text_unit: TextUnit, titles: Sequence[str]) -> str:
    return (
        f"These are clustered {text_unit.unit}s from Wikipedia. "
        "Please provide a topic summarizing and describing the cluster. "
        "The topic should cover as many of the paragraphs as reasonably possible. "
        "Make sure to look at ALL paragraphs and include them ALL in your analysis. "
        "A topic should typically be a noun phrase. "
        f"Make sure the topic you choose is NOT just one of: {'; '.join(titles)}. "
        f"{text_unit.unit}s begin here: "
    )


text_dir = Path("cache/texts")


@cache(text_dir)
def fetch_wiki(wiki_title: str, text_unit: TextUnit) -> list[str]:
    with time_segment("Wiki download", active=profile):
        page = wikipedia.page(title=wiki_title, auto_suggest=False).content
        page_body = re.sub(r"== See also ==.*", "", page, flags=re.DOTALL)
        no_headers = re.sub(r"(=+ [^=]+? =+)", "", page_body, flags=re.DOTALL)
        return tokenize(
            text_unit,
            always_bad_sentence_predicates=[],
            standalone_bad_sentence_predicates=[
                lambda sent: len(sent.split()) > 4,
            ],
            bad_para_predicates=[],
            text=no_headers,
        )


def plot_single_wiki(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    text_unit: TextUnit,
    cluster_control: ClusterControl,
    title: str,
) -> Path:
    return plot_single(
        embedding_model_name,
        wiki_embedding_instruction(text_unit),
        source=Source(title, fetch_wiki(title, text_unit)),
        dimensions=3,
        cluster_control=cluster_control,
        include_labels="include_labels",
    )


def plot_multiple_wiki(
    embedding_model_name: EmbeddingModelName[EmbeddingDim],
    text_unit: TextUnit,
    cluster_control: ClusterControl,
    titles: Sequence[str],
) -> Path:
    textss = [fetch_wiki(title, text_unit) for title in titles]
    return plot_multiple(
        embedding_model_name,
        wiki_embedding_instruction(text_unit),
        dimensions=3,
        cluster_control=cluster_control,
        sources=[Source(t, ss) for (t, ss) in zip(titles, textss, strict=True)],
        include_labels="include_labels",
    )
