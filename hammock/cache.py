from __future__ import annotations

from functools import wraps
from hashlib import sha1
from inspect import getfullargspec
import json
import pickle
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Callable,
    Any,
    Generic,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeVar,
    TypedDict,
    cast,
    overload,
)

import numpy as np

from .util import clean_filename

if TYPE_CHECKING:
    from _typeshed import SupportsWrite

P = ParamSpec("P")
T = TypeVar("T")


def _stringify(x: Any) -> str:  # pylint: disable=too-many-return-statements
    """Generate a cache key for any value we use. This is a little ad-hoc, but it works for now."""
    digest_prefix_length = 7
    match x:
        case int():
            return str(x)
        case float():
            return clean_filename(f"{x:.2f}")
        case str():
            return clean_filename(x) if len(x) < 30 else sha1(x.encode()).hexdigest()[:digest_prefix_length]
        case Path():
            return x.as_posix()
        case dict():
            return sha1("".join(str(k) + str(v) for k, v in cast(dict[Any, Any], x).items()).encode()).hexdigest()[
                :digest_prefix_length
            ]
        case list():
            return sha1("".join(str(y) for y in cast(list[Any], x)).encode()).hexdigest()[:digest_prefix_length]
        case np.ndarray():
            return sha1(cast(bytes, x)).hexdigest()[:digest_prefix_length]
        # NamedTuple
        case _ if hasattr(x, "_asdict"):
            return clean_filename("-".join(f"{k}-{_stringify(v)}" for k, v in x._asdict().items()))
        case _:
            raise ValueError(f"Cannot stringify value of type {type(x)}: {x}")


def _mk_file_name(func_name: str, *args: Any, **kwargs: Any) -> str:
    """Choose a file name for the cached result. The file name acts as a cache key and reflects
    the function name and arguments."""
    args_fragment = [clean_filename("-".join(_stringify(a) for a in args))] if args else []
    kwargs_fragment = (
        [clean_filename("-".join(f"{_stringify(k)}-{_stringify(v)}" for k, v in kwargs.items()))] if kwargs else []
    )
    return f"{'-'.join([func_name] + args_fragment + kwargs_fragment)}"


class JsonCache(TypedDict):
    format: Literal["json"]
    ext: Literal["json"]


json_cache: JsonCache = {"format": "json", "ext": "json"}


class PickleCache(TypedDict):
    format: Literal["pickle"]
    ext: Literal["pickle"]


pickle_cache: PickleCache = {"format": "pickle", "ext": "pickle"}


class BytesCache(TypedDict):
    format: Literal["bytes"]
    ext: str


CacheType = JsonCache | PickleCache | BytesCache


class Serde(NamedTuple, Generic[T]):
    load: Callable[[IO[bytes]], T]
    dump: Callable[[T, SupportsWrite[bytes | str]], None]
    # For use with `open`
    mode: Literal["b", ""]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder which also handles Numpy ints."""

    def encode(self, o: Any):
        if isinstance(o, dict):
            return super().encode(
                {
                    f"np.int64:{k}" if isinstance(k, np.int64) else k: v
                    for k, v in o.items()  # pyright: ignore[reportUnknownVariableType]
                }
            )
        else:
            return super().encode(o)

    def iterencode(self, o: Any, _one_shot: bool = False):
        if isinstance(o, dict):
            return super().iterencode(
                {
                    f"np.int64:{k}" if isinstance(k, np.int64) else k: v
                    for k, v in o.items()  # pyright: ignore[reportUnknownVariableType]
                },
                _one_shot,
            )
        else:
            return super().iterencode(o, _one_shot)


class NumpyDecoder(json.JSONDecoder):
    def decode(self, s: Any, _w: Any = ...) -> Any:
        obj = super().decode(s)
        if isinstance(obj, dict):
            return {
                (
                    np.int64(k.split(":")[1]) if isinstance(k, str) and k.startswith("np.int64:") else cast(Any, k)
                ): v
                for k, v in obj.items()  # pyright: ignore[reportUnknownVariableType]
            }
        return obj


@overload
def _format_str_to_serde(x: Literal["json"]) -> Serde[Any]:
    ...


@overload
def _format_str_to_serde(x: Literal["pickle"]) -> Serde[Any]:
    ...


@overload
def _format_str_to_serde(x: Literal["bytes"]) -> Serde[bytes]:
    ...


def _format_str_to_serde(x: Literal["json", "pickle", "bytes"]) -> Serde[Any]:
    match x:
        case "json":
            return Serde(
                lambda fp: json.load(fp, cls=NumpyDecoder),
                lambda data, handle: json.dump(data, handle, indent=2, cls=NumpyEncoder),
                "",
            )
        case "pickle":
            return Serde(pickle.load, pickle.dump, "b")
        case "bytes":

            def writer(data: bytes, handle: SupportsWrite[bytes | str]) -> None:
                """Just a little wrapper that returns `None`."""
                handle.write(data)
                return None

            return Serde(lambda handle: handle.read(), writer, "b")


class CacheResult(NamedTuple, Generic[T]):
    cache_result: T
    cache_path: Path


def cache_with_path(
    cache_dir: Path, cache_type: CacheType = pickle_cache
) -> Callable[[Callable[P, T]], Callable[P, CacheResult[T]]]:
    """Decorator to cache the result of a function call to disk. Returns result of function and path to cache
    (which is computed using the function name and arguments as cache keys).
    Note that the `bytes` format should only be used with functions that return `T = bytes`.
    Unfortunately, encoding this requirement with `@overload` is at least very hairy and maybe impossible
    due to covariance and contravariance issues."""

    def cache_decorator(func: Callable[P, T]) -> Callable[P, CacheResult[T]]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> CacheResult[T]:
            arg_names = getfullargspec(func).args
            file_name = _mk_file_name(func.__name__, **(kwargs | dict(zip(arg_names, args))))
            file_path = cache_dir / f"{file_name}.{cache_type['ext']}"
            serde = _format_str_to_serde(cache_type["format"])
            if file_path.exists():
                print(f"Reading from cache at {file_path}")
                with open(file_path, f"r{serde.mode}") as f:
                    result = cast(T, serde.load(f))
            else:
                print(f"Writing to cache at {file_path}")
                result = func(*args, **kwargs)
                if cache_type["format"] == "bytes":
                    assert type(result) == bytes
                with open(file_path, f"w{serde.mode}") as f:
                    # pyright can't figure out that the type of `result` is matched to the type of `serde.dump`
                    serde.dump(cast(Any, result), cast(Any, f))
            return CacheResult(result, file_path)

        return wrapper

    return cache_decorator


def cache(cache_dir: Path, cache_type: CacheType = pickle_cache) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to cache the result of a function call to disk. Returns result of function.
    Should be transparent to caller."""

    def cache_decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return cache_with_path(cache_dir, cache_type)(func)(*args, **kwargs).cache_result

        return wrapper

    return cache_decorator
