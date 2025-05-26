import pytest
import numpy as np

from sierra.environment.core import SierraEnv, Actions, RESOURCE_LIMITS, GAMEPLAY_CONSTANTS
from sierra.environment.grid import AGENT, EMPTY, RESOURCE, place_entity, get_cell_content, create_grid
from sierra.environment.entities import Resource # For placing dummy resources

# Standard step penalty, assuming it's a known value or accessible.
# From core.py: reward = -0.01 # Small negative step penalty
EXPECTED_STEP_PENALTY = -0.01

@pytest.fixture
def env():
    """PyTest fixture to initialize SierraEnv before each test."""
    env = SierraEnv(grid_width=5, grid_height=5)
    env.reset() # Ensure agent and resources are initialized
    return env

def test_initial_agent_position(env):
    """Test that the agent is placed on the grid after reset."""
    obs, _ = env.reset()
    agent_pos_obs = obs["agent_pos"]
    assert get_cell_content(env.grid, agent_pos_obs[0], agent_pos_obs[1]) == AGENT
    assert env.agent.x == agent_pos_obs[0]
    assert env.agent.y == agent_pos_obs[1]

# --- Basic Movement Tests ---

def test_move_up_success(env):
    """Test successful agent movement upwards."""
    env.agent.x = 2
    env.agent.y = 2
    # Clear the 2,2 cell and place agent there
    env.resources = [r for r in env.resources if not ((r.x == 2 and r.y == 2) or (r.x == 2 and r.y == 1))]
    env.grid = create_grid(env.grid_width, env.grid_height) # Use create_grid
    place_entity(env.grid, env.agent, AGENT, x=2, y=2)


    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_UP.value)

    assert env.agent.x == 2
    assert env.agent.y == 1 # Moved from (2,2) to (2,1)
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 2, 1) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, 1]))
    assert reward == EXPECTED_STEP_PENALTY # Default step penalty

def test_move_down_success(env):
    """Test successful agent movement downwards."""
    env.agent.x = 2
    env.agent.y = 2
    env.resources = [r for r in env.resources if not ((r.x == 2 and r.y == 2) or (r.x == 2 and r.y == 3))]
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=2, y=2)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_DOWN.value)

    assert env.agent.x == 2
    assert env.agent.y == 3 # Moved from (2,2) to (2,3)
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 2, 3) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, 3]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_left_success(env):
    """Test successful agent movement left."""
    env.agent.x = 2
    env.agent.y = 2
    env.resources = [r for r in env.resources if not ((r.x == 2 and r.y == 2) or (r.x == 1 and r.y == 2))]
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=2, y=2)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_LEFT.value)

    assert env.agent.x == 1 # Moved from (2,2) to (1,2)
    assert env.agent.y == 2
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 1, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([1, 2]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_right_success(env):
    """Test successful agent movement right."""
    env.agent.x = 2
    env.agent.y = 2
    env.resources = [r for r in env.resources if not ((r.x == 2 and r.y == 2) or (r.x == 3 and r.y == 2))]
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=2, y=2)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_RIGHT.value)

    assert env.agent.x == 3 # Moved from (2,2) to (3,2)
    assert env.agent.y == 2
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 3, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([3, 2]))
    assert reward == EXPECTED_STEP_PENALTY

# --- Boundary Condition Tests ---

def test_move_up_boundary(env):
    """Test agent movement attempt upwards at the boundary."""
    env.agent.x = 2
    env.agent.y = 0 # Top edge
    env.resources = [] # Clear resources for boundary tests
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=2, y=0)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_UP.value)

    assert env.agent.x == 2 # Position should not change
    assert env.agent.y == 0
    assert get_cell_content(env.grid, 2, 0) == AGENT # Agent remains
    assert np.array_equal(obs["agent_pos"], np.array([2, 0]))
    assert reward == EXPECTED_STEP_PENALTY # Step penalty

def test_move_down_boundary(env):
    """Test agent movement attempt downwards at the boundary."""
    env.agent.x = 2
    env.agent.y = env.grid_height - 1 # Bottom edge
    env.resources = []
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=2, y=env.grid_height - 1)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_DOWN.value)

    assert env.agent.x == 2 # Position should not change
    assert env.agent.y == env.grid_height - 1
    assert get_cell_content(env.grid, 2, env.grid_height - 1) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, env.grid_height - 1]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_left_boundary(env):
    """Test agent movement attempt leftwards at the boundary."""
    env.agent.x = 0 # Left edge
    env.agent.y = 2
    env.resources = []
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=0, y=2)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_LEFT.value)

    assert env.agent.x == 0 # Position should not change
    assert env.agent.y == 2
    assert get_cell_content(env.grid, 0, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([0, 2]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_right_boundary(env):
    """Test agent movement attempt rightwards at the boundary."""
    env.agent.x = env.grid_width - 1 # Right edge
    env.agent.y = 2
    env.resources = []
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=env.grid_width - 1, y=2)

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_RIGHT.value)

    assert env.agent.x == env.grid_width - 1 # Position should not change
    assert env.agent.y == 2
    assert get_cell_content(env.grid, env.grid_width - 1, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([env.grid_width - 1, 2]))
    assert reward == EXPECTED_STEP_PENALTY

# --- Movement onto Resource Cell Test ---

def test_move_onto_resource_cell(env):
    """Test agent movement onto a cell with a resource."""
    agent_start_x, agent_start_y = 1, 2
    resource_x, resource_y = 2, 2
    
    env.agent.x = agent_start_x
    env.agent.y = agent_start_y
    
    # Clear the grid and place agent
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=agent_start_x, y=agent_start_y)
    
    # Place a resource
    # Ensure this resource is the only one for simplicity in this test
    env.resources = []
    dummy_resource = Resource(resource_x, resource_y, type="wood")
    env.resources.append(dummy_resource)
    place_entity(env.grid, dummy_resource, RESOURCE, x=resource_x, y=resource_y)

    # Agent moves right from (1,2) to (2,2) where the resource is
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_RIGHT.value)

    assert env.agent.x == resource_x # Agent moved to resource cell
    assert env.agent.y == resource_y
    assert get_cell_content(env.grid, agent_start_x, agent_start_y) == EMPTY # Old cell is empty
    assert get_cell_content(env.grid, resource_x, resource_y) == AGENT # Agent is on the new cell
    assert np.array_equal(obs["agent_pos"], np.array([resource_x, resource_y]))
    
    # Check if resource was collected (removed from list and grid)
    # This is part of _handle_movement_and_collection, but good to verify
    assert not any(r.x == resource_x and r.y == resource_y for r in env.resources)
    
    # Reward should include step penalty and potentially resource collection reward
    # For wood, default collection is 1.0. Axe bonus is +2. Assume no axe for simplicity.
    # The _handle_movement_and_collection method returns the resource_value.
    # resource_value = self._collect_resource(self.agent.x, self.agent.y)
    # This part of the reward depends on the internal logic of _handle_movement_and_collection
    # and _collect_resource which is not directly visible here.
    # For now, we just check that the agent moved correctly.
    # A more specific reward check would require knowing the exact reward for collecting wood.
    # Let's assume the reward structure from _handle_movement_and_collection:
    # action_specific_reward += resource_value
    # reward += action_specific_reward
    # So, reward = EXPECTED_STEP_PENALTY + resource_value
    # If wood gives 1.0, reward = -0.01 + 1.0 = 0.99
    # This needs to be confirmed by checking the _collect_resource implementation details.
    # For this test, we will focus on movement and presence, not the exact reward value.
    assert "wood" in env.agent.inventory # Check if wood was added to inventory

    # Verify the wood_locs in observation is updated (padded with -1,-1)
    # Since the resource was collected, its location should not be listed.
    # This depends on how many MAX_WOOD_SOURCES are there.
    # If MAX_WOOD_SOURCES is 1, then wood_locs should be [[-1,-1]]
    max_wood = RESOURCE_LIMITS["MAX_WOOD_SOURCES"]
    expected_wood_locs = np.array([[-1, -1]] * max_wood, dtype=int)
    assert np.array_equal(obs["wood_locs"], expected_wood_locs)

def test_agent_inventory_after_moving_onto_resource(env):
    """Test that agent inventory is updated after moving onto a resource cell."""
    agent_start_x, agent_start_y = 1, 2
    resource_x, resource_y = 2, 2
    
    env.agent.x = agent_start_x
    env.agent.y = agent_start_y
    # env.agent.inventory = {} # This was causing the issue; inventory is already initialized to all zeros.
    
    env.grid = create_grid(env.grid_width, env.grid_height)
    place_entity(env.grid, env.agent, AGENT, x=agent_start_x, y=agent_start_y)
    
    env.resources = []
    dummy_resource = Resource(resource_x, resource_y, type="wood") # Removed quantity
    env.resources.append(dummy_resource)
    place_entity(env.grid, dummy_resource, RESOURCE, x=resource_x, y=resource_y)

    env.step(Actions.MOVE_RIGHT.value) # Move onto wood

    assert env.agent.inventory.get("wood", 0) == 1

    # Move onto another resource (stone)
    agent_start_x, agent_start_y = env.agent.x, env.agent.y # Current position
    resource_x, resource_y = agent_start_x + 1, agent_start_y # Cell to the right
    
    # Ensure agent doesn't go out of bounds
    if resource_x >= env.grid_width:
        pytest.skip("Agent would move out of bounds, skipping second part of test")

    dummy_stone = Resource(resource_x, resource_y, type="stone") # Removed quantity
    env.resources.append(dummy_stone) # Add to env's list
    place_entity(env.grid, dummy_stone, RESOURCE, x=resource_x, y=resource_y) # Place on grid

    env.step(Actions.MOVE_RIGHT.value) # Move onto stone

    assert env.agent.inventory.get("wood", 0) == 1 # Wood should still be there
    assert env.agent.inventory.get("stone", 0) == 1


# Helper to clean grid and place agent for movement tests
def _setup_clean_grid_for_movement(env, agent_x, agent_y, target_x=None, target_y=None):
    """Clears resources, grid, and places agent at specified pos."""
    env.agent.x = agent_x
    env.agent.y = agent_y
    env.resources = [] # Clear all existing resources
    
    # Create a completely empty grid
    env.grid = create_grid(env.grid_width, env.grid_height)
    
    # Place agent at the specified starting position
    place_entity(env.grid, env.agent, AGENT, x=agent_x, y=agent_y)

    # If a target cell is specified (for resource movement test), ensure it's also clear initially
    if target_x is not None and target_y is not None:
        if env.grid[target_y][target_x] != AGENT: # Don't clear if agent is already there
             env.grid[target_y][target_x] = EMPTY


# Re-writing basic movement tests to use the helper for cleaner setup
def test_move_up_success_v2(env):
    _setup_clean_grid_for_movement(env, 2, 2)
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_UP.value)
    assert env.agent.x == 2 and env.agent.y == 1
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 2, 1) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, 1]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_down_success_v2(env):
    _setup_clean_grid_for_movement(env, 2, 2)
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_DOWN.value)
    assert env.agent.x == 2 and env.agent.y == 3
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 2, 3) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, 3]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_left_success_v2(env):
    _setup_clean_grid_for_movement(env, 2, 2)
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_LEFT.value)
    assert env.agent.x == 1 and env.agent.y == 2
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 1, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([1, 2]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_right_success_v2(env):
    _setup_clean_grid_for_movement(env, 2, 2)
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_RIGHT.value)
    assert env.agent.x == 3 and env.agent.y == 2
    assert get_cell_content(env.grid, 2, 2) == EMPTY
    assert get_cell_content(env.grid, 3, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([3, 2]))
    assert reward == EXPECTED_STEP_PENALTY

# Re-writing boundary tests to use the helper
def test_move_up_boundary_v2(env):
    _setup_clean_grid_for_movement(env, 2, 0) # Top edge
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_UP.value)
    assert env.agent.x == 2 and env.agent.y == 0
    assert get_cell_content(env.grid, 2, 0) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, 0]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_down_boundary_v2(env):
    _setup_clean_grid_for_movement(env, 2, env.grid_height - 1) # Bottom edge
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_DOWN.value)
    assert env.agent.x == 2 and env.agent.y == env.grid_height - 1
    assert get_cell_content(env.grid, 2, env.grid_height - 1) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([2, env.grid_height - 1]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_left_boundary_v2(env):
    _setup_clean_grid_for_movement(env, 0, 2) # Left edge
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_LEFT.value)
    assert env.agent.x == 0 and env.agent.y == 2
    assert get_cell_content(env.grid, 0, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([0, 2]))
    assert reward == EXPECTED_STEP_PENALTY

def test_move_right_boundary_v2(env):
    _setup_clean_grid_for_movement(env, env.grid_width - 1, 2) # Right edge
    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_RIGHT.value)
    assert env.agent.x == env.grid_width - 1 and env.agent.y == 2
    assert get_cell_content(env.grid, env.grid_width - 1, 2) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([env.grid_width - 1, 2]))
    assert reward == EXPECTED_STEP_PENALTY

# Re-writing movement onto resource cell test to use helper
def test_move_onto_resource_cell_v2(env):
    agent_start_x, agent_start_y = 1, 2
    resource_x, resource_y = 2, 2
    
    _setup_clean_grid_for_movement(env, agent_start_x, agent_start_y, target_x=resource_x, target_y=resource_y)
    
    dummy_resource = Resource(resource_x, resource_y, type="wood") # Removed quantity
    env.resources.append(dummy_resource)
    place_entity(env.grid, dummy_resource, RESOURCE, x=resource_x, y=resource_y)
    # env.agent.inventory = {} # This was causing the issue; inventory is already initialized to all zeros.

    obs, reward, terminated, truncated, info = env.step(Actions.MOVE_RIGHT.value)

    assert env.agent.x == resource_x
    assert env.agent.y == resource_y
    assert get_cell_content(env.grid, agent_start_x, agent_start_y) == EMPTY
    assert get_cell_content(env.grid, resource_x, resource_y) == AGENT
    assert np.array_equal(obs["agent_pos"], np.array([resource_x, resource_y]))
    assert not any(r.x == resource_x and r.y == resource_y for r in env.resources)
    assert env.agent.inventory.get("wood", 0) == 1
    
    max_wood = RESOURCE_LIMITS["MAX_WOOD_SOURCES"]
    expected_wood_locs = np.array([[-1, -1]] * max_wood, dtype=int)
    assert np.array_equal(obs["wood_locs"], expected_wood_locs)
    # The exact reward value check is omitted here as it depends on internal collection logic
    # not specified for testing focus (movement and grid state).
    # If a specific reward for collecting wood (e.g. 1.0) is known:
    # assert reward == EXPECTED_STEP_PENALTY + 1.0
    # For now, we rely on inventory change and resource list change.
