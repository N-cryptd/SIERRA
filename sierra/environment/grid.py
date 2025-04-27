import numpy as np

# Define constants for cell types
EMPTY = 0
WALL = 1
AGENT = 2
RESOURCE = 3

def create_grid(width, height):
    """Creates a new grid of specified dimensions."""
    return np.full((height, width), EMPTY, dtype=int)

def place_entity(grid, entity, cell_type):
    """Places an entity (agent or resource) on the grid."""
    if check_boundaries(grid, entity.x, entity.y):
        grid[entity.y, entity.x] = cell_type
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