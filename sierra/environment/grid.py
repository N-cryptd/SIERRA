"""Utility helpers for working with the grid representation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

EMPTY = 0
AGENT = 1
RESOURCE = 2
THREAT = 3

PLAINS = 0
FOREST = 1
ROCKY = 2


def create_grid(width: int, height: int) -> np.SimpleArray:
    return np.full((height, width), EMPTY, dtype=int)


def check_boundaries(grid: np.SimpleArray, x: int, y: int) -> bool:
    height, width = grid.shape
    return 0 <= x < width and 0 <= y < height


def place_entity(grid: np.SimpleArray, entity, cell_type: int, x: int | None = None, y: int | None = None) -> bool:
    target_x = entity.x if x is None else x
    target_y = entity.y if y is None else y
    if not check_boundaries(grid, target_x, target_y):
        return False
    grid[target_y, target_x] = cell_type
    entity.x, entity.y = target_x, target_y
    return True


def get_cell_content(grid: np.SimpleArray, x: int, y: int) -> int | None:
    if not check_boundaries(grid, x, y):
        return None
    return grid[y, x]


def iter_coordinates(width: int, height: int) -> Iterable[Tuple[int, int]]:
    for y in range(height):
        for x in range(width):
            yield (x, y)


__all__ = [
    "EMPTY",
    "AGENT",
    "RESOURCE",
    "THREAT",
    "PLAINS",
    "FOREST",
    "ROCKY",
    "create_grid",
    "place_entity",
    "check_boundaries",
    "get_cell_content",
    "iter_coordinates",
]
