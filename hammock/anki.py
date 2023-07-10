import argparse
from pathlib import Path
import re
import sqlite3
from typing import Mapping, NamedTuple, NewType

import html2text

from hammock.plot import Source

from .core import plot_single
from .cluster import CCColorAndSummarize
from .embedding import instructor_large


def strip_html(text: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_tables = True
    h.ignore_emphasis = True
    # Do it twice because some cards about HTML turn into valid HTML after the first pass!
    return h.handle(h.handle(text)).strip()


class StripClozeResults(NamedTuple):
    text: str
    had_cloze_deletions: bool


def strip_cloze_deletions(text: str) -> StripClozeResults:
    out = re.sub(r"{{c\d+::(.*?)(::.*?)?}}", r"\1", text, flags=re.DOTALL)
    return StripClozeResults(out, text != out)


def compress_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


NoteID = NewType("NoteID", int)


def extract_text_from_anki(private_path: Path, db_path: Path) -> Mapping[NoteID, str]:
    with open(private_path) as f:
        private_topics = [line.strip() for line in f.readlines()]
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, flds FROM notes")
        field_separator = "\x1f"
        return {
            NoteID(note_id): stripped.text
            for note_id, stripped in [
                (row[0], strip_cloze_deletions(compress_spaces(strip_html(row[1].split(field_separator)[0]))))
                for row in cursor.fetchall()
            ]
            if stripped.had_cloze_deletions
            # and "<script" not in t
            and len(stripped.text.split()) > 4 and all(sub not in stripped.text for sub in private_topics)
        }


summary_prompt = (
    "Please choose an academic subfield as topic to label the following cluster of snippets.  "
    "The topic should cover as many of the snippets as reasonably possible. "
    "Make sure to look at ALL snippets and include them ALL in your analysis. "
    "A topic should typically be a noun phrase. "
    "Snippets begin here: "
)
embedding_instruction = "Represent the Academic paragraph for clustering by topic: "

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Little helper for generating plots of Anki cards.")
    parser.add_argument("-d", "--db-path", type=Path, help="Path to Anki database.")
    parser.add_argument(
        "-p",
        "--private-path",
        type=Path,
        help=(
            "Path to file containing private topics to exclude from visualization. "
            "Format is one substring per line."
        ),
    )
    args = parser.parse_args()
    notes = extract_text_from_anki(args.private_path, args.db_path)
    plot_single(
        instructor_large,
        embedding_instruction,
        dimensions=3,
        cluster_control=CCColorAndSummarize("google/flan-t5-large", summary_prompt, [60, 30]),
        source=Source("Anki Cards", list(notes.values())),
        include_labels="include_labels",
    )
