"""Utility helpers for the grid based SIERRA environment."""

from __future__ import annotations

import numpy as np

# Cell encodings used throughout the environment
EMPTY: int = 0
WALL: int = 1
AGENT: int = 2
RESOURCE: int = 3
THREAT: int = 4


def create_grid(width: int, height: int) -> np.ndarray:
    """Create a new grid filled with :data:`EMPTY` cells."""
    return np.full((height, width), EMPTY, dtype=int)


def place_entity(
    grid: np.ndarray,
    entity_or_cell_type,
    cell_type: int | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
) -> None:
    """Place an entity on the grid at the requested coordinates.

    The helper supports both the historical signature
    ``place_entity(grid, entity, cell_type, x=..., y=...)`` used in the tests
    and the simplified ``place_entity(grid, cell_type, x=..., y=...)`` variant
    used internally by the environment.
    """

    if cell_type is None:
        cell_type = int(entity_or_cell_type)
        target_x = x
        target_y = y
    else:
        target_x = x if x is not None else getattr(entity_or_cell_type, "x")
        target_y = y if y is not None else getattr(entity_or_cell_type, "y")

    if target_x is None or target_y is None:
        raise ValueError("Target coordinates must be provided for place_entity")

    grid[target_y, target_x] = int(cell_type)


def check_boundaries(grid: np.ndarray, x: int, y: int) -> bool:
    """Return ``True`` when ``(x, y)`` is inside the grid boundaries."""
    height, width = grid.shape
    return 0 <= x < width and 0 <= y < height


def get_cell_content(grid: np.ndarray, x: int, y: int) -> int | None:
    """Return the cell value for ``(x, y)`` if it is in bounds."""
    if check_boundaries(grid, x, y):
        return int(grid[y, x])
    return None
