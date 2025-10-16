"""Minimal stand-in for the parts of Gymnasium used in the tests."""

from __future__ import annotations

import random as _random
from types import SimpleNamespace
from typing import Any, Dict as TypingDict


class SimpleRandom:
    def __init__(self):
        self._rng = _random.Random()

    def seed(self, seed: int | None = None):
        self._rng.seed(seed)

    def choice(self, items):
        seq = list(items)
        if not seq:
            raise IndexError("Cannot choose from an empty sequence")
        return self._rng.choice(seq)

    def shuffle(self, seq):
        self._rng.shuffle(seq)


class Env:
    metadata: TypingDict[str, Any] = {}

    def __init__(self):
        self.np_random = SimpleRandom()

    def reset(self, seed: int | None = None, options: Any | None = None):
        if seed is not None:
            self.np_random.seed(seed)
        return None, {}

    def step(self, action):
        raise NotImplementedError

    def render(self, mode: str = "human"):
        return None

    def close(self):
        return None


class Box:
    def __init__(self, low, high, shape=None, dtype=float):
        self.low = low
        self.high = high
        self.shape = tuple(shape) if shape is not None else (1,)
        self.dtype = dtype

    def sample(self):
        return 0


class Discrete:
    def __init__(self, n: int):
        self.n = int(n)

    def sample(self) -> int:
        return _random.randint(0, self.n - 1) if self.n > 0 else 0


class Dict:
    def __init__(self, spaces: TypingDict[str, Any]):
        self.spaces = spaces

    def sample(self):
        return {key: space.sample() if hasattr(space, "sample") else None for key, space in self.spaces.items()}


spaces = SimpleNamespace(Box=Box, Discrete=Discrete, Dict=Dict)

__all__ = ["Env", "spaces"]
