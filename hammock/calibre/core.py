import curses
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Literal, Mapping, Sequence, TypedDict, cast

os.environ["OMP_NUM_THREADS"] = "4"

from nltk.tokenize import sent_tokenize, word_tokenize

from ..cache import cache, json_cache
from ..cluster import CCColorAndSummarize
from ..core import TextUnit, clip_string, para_tokenize, plot_single, tokenize
from ..embedding import instructor_large
from ..plot import Source
from ..util import clean_filename
from .detect_core import get_core


class RawBook(TypedDict):
    title: str
    authors: str
    rating: int
    formats: list[str]


class EpubBook(TypedDict):
    title: str
    authors: str
    rating: int
    path: Path


def get_books_list(library_path: Path) -> Sequence[EpubBook]:
    """Fetch collection of epubs known by Calibre."""
    result = subprocess.run(
        [
            "calibredb",
            "list",
            "--library-path",
            library_path,
            "--for-machine",
            "--fields",
            "title,authors,rating,formats",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    raw: list[RawBook] = json.loads(result.stdout)
    epubs = [book for book in raw if any(path.endswith("epub") for path in book["formats"])]
    return [
        cast(
            EpubBook, {**book, "path": Path(next(iter(path for path in book["formats"] if path.endswith("epub"))))}
        )
        for book in epubs
    ]


error_list: list[Path] = []


def convert_book_to_text(book_path: Path, output_dir: Path):
    convert_destination = output_dir / f"{clean_filename(book_path.stem)}.txt"
    if not convert_destination.exists():
        res = subprocess.run(["ebook-convert", book_path.as_posix(), convert_destination.as_posix()], check=False)
        if res.returncode != 0:
            error_list.append(book_path)


def convert_epubs():
    books = get_books_list(Path(os.path.expanduser("~/calibre-library")))
    for book in books:
        convert_book_to_text(book["path"], Path("cache/calibre-conversions"))
    for error in error_list:
        print(error)


# A = TypeVar("A")
def summary_prompt(book_descr: str, embedding_area: str) -> str:
    return (
        "These are clustered paragraphs from a book. "
        # f"These are clustered paragraphs from {book_descr}. "
        "Please provide a topic summarizing and describing the cluster. "
        # "The topic should cover as many of the paragraphs as reasonably possible. "
        # "Make sure to look at ALL paragraphs and include them ALL in your analysis. "
        # f"Your response should NOT just be {embedding_area}. "
        # "A topic should typically be a noun phrase. "
        "Paragraphs begin here: "
    )


def handle_book(
    book_path: Path, start_anchor: str, end_anchor: str, title: str, embedding_area: str, summary_book_descr: str
):
    """Produce visualization for given book."""
    with open(book_path, encoding="utf-8") as f:
        text = f.read()
    core_text = clip_string(text, start_anchor=start_anchor, end_anchor=end_anchor)
    assert core_text is not None
    paras = tokenize(
        TextUnit("paragraph", para_newline_type="double"),
        always_bad_sentence_predicates=[
            lambda sent: sent.startswith("Source"),
        ],
        standalone_bad_sentence_predicates=[
            lambda sent: bool(re.search(r"^\d+\.", sent)),  # Corresponds to questions usually
            lambda sent: len(sent.split()) < 4,
            lambda sent: bool(re.search(r" p. ", sent)),
            lambda sent: bool(re.search(r" pp. ", sent)),
        ],
        bad_para_predicates=[
            lambda para: all(s.endswith("?") for s in sent_tokenize(para)),
            lambda para: para.lower().startswith("chapter") or para.lower().startswith("section"),
            lambda para: para.lower().startswith("table") or para.lower().startswith("figure"),
            lambda para: para.lower().startswith("exercise"),
            lambda para: len(word_tokenize(para)) < 12,
            lambda para: len(word_tokenize(para)) < 24
            and (bool(re.search(r" p. ", para)) or bool(re.search(r" pp. ", para))),
        ],
        text=core_text,
    )
    return plot_single(
        instructor_large,
        f"Represent the {embedding_area} paragraph for clustering: ",
        dimensions=3,
        # cluster_control=CCColor(min_cluster_sizes=[20, 5]),
        cluster_control=CCColorAndSummarize(
            "google/flan-t5-large", summary_prompt(summary_book_descr, embedding_area), [20, 12, 8, 5]
        ),
        source=Source(title, paras),
        include_labels="include_labels",
    )


BookArgs = TypedDict(
    "BookArgs",
    {
        "start_anchor": str,
        "end_anchor": str,
        "title": str,
        "embedding_area": str,
        "summary_book_descr": str,
    },
)


def choose_snippet(snippets: list[str]) -> str:
    """Interactively choose text as start or end anchor demarcating pre- and postamble from main content."""
    selected_index = 0

    stdscr = curses.initscr()
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    win = curses.newwin(curses.LINES, curses.COLS, 0, 0)

    try:
        while True:
            win.erase()

            win.addstr(0, 0, "Choose a snippet with j/J/k/K, press q to finalize")

            # Display snippets with the selected snippet highlighted
            context_length = 25
            start_index = max(0, selected_index - context_length)
            for i, s in enumerate(
                snippets[
                    start_index : min(len(snippets), max(selected_index + context_length, 2 * context_length))
                ],
                start_index,
            ):
                if i == selected_index:
                    win.attron(curses.A_REVERSE)
                win.addstr(i - start_index + 1, 0, s)
                win.attroff(curses.A_REVERSE)

            win.refresh()

            key = stdscr.getch()

            if key == ord("q"):
                curses.endwin()
                return snippets[selected_index]
            elif key == ord("k") and selected_index > 0:
                selected_index -= 1
            elif key == ord("K") and selected_index > 50:
                selected_index -= 50
            elif key == ord("j") and selected_index < len(snippets) - 1:
                selected_index += 1
            elif key == ord("J") and selected_index < len(snippets) - 50:
                selected_index += 50
    finally:
        curses.endwin()


book_args_dir = Path("cache/book-args")


@cache(book_args_dir, cache_type=json_cache)
def collect_book_args(anchor_method: Literal["manual", "automatic"], book_path: Path, title: str) -> BookArgs:
    match anchor_method:
        case "automatic":
            start_anchor, end_anchor = get_core(book_path)
            pruned_title = re.sub(r"\([^)]*\)", "", title).split(":")[0]
            embedding_area = pruned_title
            summary_book_descr = "a book titled " + pruned_title
        case "manual":
            with open(book_path, encoding="utf-8") as f:
                text = f.read()
            start_anchor = choose_snippet(para_tokenize("double", text))
            end_anchor = choose_snippet(list(reversed(para_tokenize("double", text))))
            print("Start: ", repr(start_anchor))
            print("End: ", repr(end_anchor))
            print("Title: ", title)
            embedding_area = input("Embedding area (e.g. 'Anthropology'): ")
            summary_book_descr = input("Summary book description (e.g. 'an anthropology textbook'): ")

    return BookArgs(
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        title=title,
        embedding_area=embedding_area,
        summary_book_descr=summary_book_descr,
    )


completed: Mapping[str, str] = {}


def run_epub(anchor_method: Literal["manual", "automatic"], book_path: Path, title: str):
    book_args = collect_book_args(anchor_method, book_path, title)
    completed[title] = handle_book(book_path, **book_args).as_posix()
    with open("output/calibre.json", "w") as f:
        json.dump(completed, f, indent=2)


def paths_to_titles() -> Callable[[Path], str]:
    """Convert a path to a title by looking up the path in the calibre library.
    Return a `Callable` so that we keep the book list in memory instead of
    shelling out and hitting the disk each time."""
    books = get_books_list(Path(os.path.expanduser("~/calibre-library")))
    return lambda x: next(book["title"] for book in books if clean_filename(book["path"].stem) == x.stem)


# Batch process books from Calibre library
if __name__ == "__main__":
    ptt = paths_to_titles()
    for book_path in Path("cache/calibre-conversions").glob("*Very-Short*.txt"):
        run_epub("automatic", book_path, ptt(book_path))
