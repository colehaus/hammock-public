from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import re
import time
from typing import (
    Any,
    Callable,
    Generic,
    Sequence,
    TypeAlias,
    TypeVar,
    overload,
)

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def clean_filename(filename: str) -> str:
    return re.sub(r'[\s./\*\?\[\]\'"|(){}<>!@#$%^&:;~`_,;]+', "-", filename)


def declare(type_: type[A], value: A) -> A:
    """Declare a type inline"""
    return value


@overload
def unzip(x: Sequence[tuple[A, B, C]]) -> tuple[Sequence[A], Sequence[B], Sequence[C]]:
    ...


@overload
def unzip(x: Sequence[tuple[A, B]]) -> tuple[Sequence[A], Sequence[B]]:
    ...


def unzip(x: Sequence[tuple[Any, ...]]) -> tuple[Sequence[Any], ...]:
    return tuple(zip(*x))


@contextmanager
def time_segment(name: str, active: bool = True):
    """Quick and dirty profiling helper. Reports time spent in block."""
    if active:
        print(f"Entering {name}")
        ts = time.time()
        yield
        print(f"Exiting {name} after {timedelta(seconds=time.time() - ts)}")
    else:
        yield


def transpose(lists: Sequence[Sequence[A]]) -> Sequence[Sequence[A]]:
    return [list(sub_list) for sub_list in zip(*lists)]


def thunkify(fn: Callable[[], A]) -> Callable[[], A]:
    """Memoize a function with no arguments"""
    x = None

    def inner():
        nonlocal x
        if x is None:
            x = fn()
            return x
        else:
            return x

    return inner


X_co = TypeVar("X_co", covariant=True)
Y_co = TypeVar("Y_co", covariant=True)


@dataclass(frozen=True)
class Failure(Generic[X_co]):
    value: X_co


@dataclass(frozen=True)
class Success(Generic[X_co]):
    value: X_co


Either: TypeAlias = Failure[X_co] | Success[Y_co]


def flatmap(fn: Callable[[A], Sequence[B]], input_list: Sequence[A]) -> Sequence[B]:
    return [item for sublist in (fn(x) for x in input_list) for item in sublist]


def flatten(x: Sequence[Sequence[A]]) -> list[A]:
    return [item for sublist in x for item in sublist]


def unstack(x: Sequence[A], indices: list[int]) -> Sequence[Sequence[A]]:
    """Split a sequence into subsequences at the given indices"""
    return [x[start:end] for start, end in zip([0] + indices, indices + [len(x)])]
