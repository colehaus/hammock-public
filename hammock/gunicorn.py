from typing import Any

from gunicorn.app.wsgiapp import WSGIApplication


class GunicornApp(WSGIApplication):
    def init(self, parser: Any, opts: Any, args: list[Any]):
        # Tell gunicorn which app to run
        # Like running `gunicorn hammock.web:app` from the command line
        args.insert(0, "hammock.web:app")
        super().init(parser, opts, args)
        self.cfg.set("timeout", 600)
        # Auto-reload
        self.cfg.set("reload", True)


# Need this as a separate function so that it can be specified in
# the `tool.poetry.scripts` section of `pyproject.toml`
def main():
    GunicornApp().run()


if __name__ == "__main__":
    main()
