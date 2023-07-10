# pylint: skip-file

class WikipediaPage:
    content: str

def page(title: str, auto_suggest: bool = True) -> WikipediaPage: ...
