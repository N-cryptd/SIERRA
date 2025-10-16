"""A lightweight subset of NumPy used for the kata tests.

This module implements only the minimal functionality required by the
unit tests.  It is *not* a drop-in replacement for NumPy.
"""

from __future__ import annotations

from math import prod as _math_prod
import random as _random
from typing import Any, Iterable, Sequence, Tuple, Union

Number = Union[int, float]

float32 = float
bool_ = bool


def _coerce_value(value: Any, dtype: Any | None) -> Any:
    if dtype is None:
        return value
    try:
        return dtype(value)
    except Exception:
        return value


def _deep_copy(data: Any) -> Any:
    if isinstance(data, list):
        return [_deep_copy(v) for v in data]
    return data


def _ensure_nested_lists(data: Any, dtype: Any | None) -> Any:
    if isinstance(data, SimpleArray):
        return _deep_copy(data._data)
    if isinstance(data, (list, tuple)):
        return [_ensure_nested_lists(v, dtype) for v in data]
    return _coerce_value(data, dtype)


def _infer_shape(data: Any) -> Tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        inner_shape = _infer_shape(data[0]) if isinstance(data[0], list) else ()
        return (len(data),) + inner_shape
    return ()


def _flatten(data: Any) -> list[Any]:
    if isinstance(data, list):
        result: list[Any] = []
        for item in data:
            result.extend(_flatten(item))
        return result
    return [data]


class SimpleArray:
    """Minimal array type supporting the operations used in the tests."""

    def __init__(self, data: Any, dtype: Any | None = None):
        self.dtype = dtype
        self._data = _ensure_nested_lists(data, dtype)

    @property
    def shape(self) -> Tuple[int, ...]:
        return _infer_shape(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            target = self._data
            for part in key:
                target = target[part]
            return target
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            target = self._data
            for part in key[:-1]:
                target = target[part]
            target[key[-1]] = _coerce_value(value, self.dtype)
        else:
            self._data[key] = _coerce_value(value, self.dtype)

    def flatten(self) -> list[Any]:
        return _flatten(self._data)

    def tolist(self) -> list[Any]:
        return _deep_copy(self._data)

    def __repr__(self) -> str:
        return f"SimpleArray({self._data})"


def array(data: Any, dtype: Any | None = None) -> SimpleArray:
    return SimpleArray(data, dtype=dtype)


def zeros(shape: Sequence[int], dtype: Any | None = int) -> SimpleArray:
    if isinstance(shape, int):
        shape = (shape,)
    def build(level: int) -> Any:
        if level == len(shape) - 1:
            return [_coerce_value(0, dtype) for _ in range(shape[level])]
        return [build(level + 1) for _ in range(shape[level])]
    data = build(0) if shape else _coerce_value(0, dtype)
    return SimpleArray(data, dtype=dtype)


def full(shape: Sequence[int], fill_value: Any, dtype: Any | None = None) -> SimpleArray:
    if isinstance(shape, int):
        shape = (shape,)
    def build(level: int) -> Any:
        if level == len(shape) - 1:
            return [_coerce_value(fill_value, dtype) for _ in range(shape[level])]
        return [build(level + 1) for _ in range(shape[level])]
    data = build(0) if shape else _coerce_value(fill_value, dtype)
    return SimpleArray(data, dtype=dtype)


def array_equal(a: Any, b: Any) -> bool:
    return _to_list(a) == _to_list(b)


def _to_list(value: Any) -> Any:
    if isinstance(value, SimpleArray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_list(v) for v in value]
    return value


def prod(values: Iterable[int]) -> int:
    return _math_prod(values)


def concatenate(values: Iterable[Any], axis: int = 0) -> SimpleArray:
    lists = [_to_list(v) for v in values]
    result: list[Any] = []
    for item in lists:
        result.extend(item)
    return array(result)


def mean(values: Iterable[Number]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def std(values: Iterable[Number]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    m = mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def clip(values: Iterable[Number], min_value: Number, max_value: Number) -> SimpleArray:
    return array([min(max(v, min_value), max_value) for v in values])


def isscalar(value: Any) -> bool:
    return not isinstance(value, (list, tuple, SimpleArray))


class _RandomModule:
    def __init__(self):
        self._rng = _random.Random()

    def seed(self, seed: int | None = None):
        self._rng.seed(seed)

    def choice(self, a: Sequence[Any], size: Sequence[int] | int | None = None):
        seq = list(a)
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        if size is None:
            return self._rng.choice(seq)
        if isinstance(size, int):
            size = (size,)
        def build(level: int) -> Any:
            if level == len(size) - 1:
                return [_coerce_value(self._rng.choice(seq), None) for _ in range(size[level])]
            return [build(level + 1) for _ in range(size[level])]
        return array(build(0))

    def shuffle(self, x: list[Any]):
        self._rng.shuffle(x)


random = _RandomModule()

__all__ = [
    "SimpleArray",
    "array",
    "array_equal",
    "zeros",
    "full",
    "prod",
    "concatenate",
    "mean",
    "std",
    "clip",
    "float32",
    "random",
]
