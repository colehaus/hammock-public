import json
from flask import Flask, render_template, redirect, request, url_for
from flask_compress import Compress

from .core import (
    GutenbergArgs,
    TextUnit,
    plot_multiple_gutenberg,
    plot_multiple_wiki,
    plot_freeform,
    plot_single_gutenberg,
    plot_single_wiki,
)
from .cluster import CCColor, SummaryModelName
from .embedding import instructor_base
from .util import Failure, Success

app = Flask(__name__, template_folder="../templates", static_folder="../output")
Compress(app)


@app.route("/books", methods=["GET"])
def books():
    with open("output/calibre.json", "r") as f:
        return render_template("books.html", books=json.load(f))


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/freeform", methods=["GET", "POST"])
def freeform():
    match request.method:
        case "POST":
            return redirect(
                plot_freeform(
                    default_embedding_model,
                    "Represent the text for clustering:",
                    request.form["input_field"].splitlines(),
                ).as_posix()
            )
        case "GET":
            return render_template(
                "main.html",
                title="custom sentences",
                label="Sentences separated by newlines (minimum of 7)",
                placeholder="Sphinx of black quartz, judge my vow.",
                action=url_for("freeform"),
            )
        case method:
            raise ValueError(f"Unexpected method: {method}")


@app.route("/gutenberg", methods=["GET", "POST"])
def gutenberg():
    match request.method:
        case "POST":
            match [
                GutenbergArgs(*s) for s in [line.split("|") for line in request.form["input_field"].splitlines()]
            ]:
                case [gutenberg_args]:
                    match plot_single_gutenberg(
                        default_embedding_model,
                        TextUnit("paragraph", para_newline_type="double"),
                        CCColor(min_cluster_sizes=[20, 5]),
                        gutenberg_args.title,
                        gutenberg_args.start_anchor,
                        gutenberg_args.end_anchor,
                    ):
                        case Failure(err):
                            raise ValueError(err)
                        case Success(out_file):
                            return redirect(out_file.as_posix())
                case gutenberg_argss:
                    return redirect(
                        plot_multiple_gutenberg(
                            default_embedding_model,
                            TextUnit("paragraph", para_newline_type="double"),
                            CCColor(min_cluster_sizes=[20, 5]),
                            gutenberg_argss,
                        ).as_posix()
                    )
        case "GET":
            return render_template(
                "main.html",
                title="Project Gutenberg books",
                label="Exact title of Project Gutenberg books (one per line (or just one)."
                "<br> Include a fragment after '|' to start with that fragment"
                " and another fragment after another '|' to mark the end.",
                placeholder=(
                    "Frankenstein; Or, The Modern Prometheus|"
                    "You will rejoice to hear that no disaster|lost in darkness and distance."
                ),
                action=url_for("gutenberg"),
            )
        case method:
            raise ValueError(f"Unexpected method: {method}")


default_embedding_model = instructor_base
default_summary_model: SummaryModelName = "google/flan-t5-base"


@app.route("/wiki", methods=["GET", "POST"])
def wiki():
    match request.method:
        case "POST":
            match request.form["input_field"].splitlines():
                case [title]:
                    return redirect(
                        plot_single_wiki(
                            default_embedding_model,
                            TextUnit("paragraph", para_newline_type="single"),
                            CCColor(
                                min_cluster_sizes=[8, 3],
                            ),
                            title,
                        ).as_posix()
                    )
                case titles:
                    return redirect(
                        plot_multiple_wiki(
                            default_embedding_model,
                            TextUnit("paragraph", para_newline_type="single"),
                            CCColor(min_cluster_sizes=[8, 3]),
                            titles,
                        ).as_posix()
                    )
        case "GET":
            return render_template(
                "main.html",
                title="Wikipedia articles",
                label="Exact title of Wikipedia articles (one per line (or just one))",
                placeholder="New York City",
                action=url_for("wiki"),
            )
        case method:
            raise ValueError(f"Unexpected method: {method}")


if __name__ == "__main__":
    app.run()
