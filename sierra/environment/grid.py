import numpy as np

# Define constants for cell types
EMPTY = 0
WALL = 1
AGENT = 2
RESOURCE = 3
THREAT = 4

# Terrain Types
PLAINS = 0
FOREST = 1
ROCKY = 2

def create_grid(width, height):
    """Creates a new grid of specified dimensions."""
    return np.full((height, width), EMPTY, dtype=int)

def place_entity(grid, entity, cell_type, x=None, y=None):
    """Places an entity (agent or resource) on the grid.
    If x and y are provided, they are used as the entity's new coordinates.
    Otherwise, entity.x and entity.y are used.
    """
    target_x = x if x is not None else entity.x
    target_y = y if y is not None else entity.y

    if check_boundaries(grid, target_x, target_y):
        grid[target_y, target_x] = cell_type
        return True
    return False

def check_boundaries(grid, x, y):
    """Checks if the given coordinates are within grid boundaries."""
    height, width = grid.shape
    return 0 <= x < width and 0 <= y < height

def get_cell_content(grid, x, y):
    """Gets the content of the cell at the given coordinates."""
    if check_boundaries(grid, x, y):
        return grid[y, x]
    return None # Or raise an error, depending on desired behavior