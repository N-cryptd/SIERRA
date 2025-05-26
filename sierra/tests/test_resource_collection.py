import pytest
import numpy as np

from sierra.environment.core import (
    SierraEnv, Actions, RESOURCE, EMPTY, AGENT,
    GAMEPLAY_CONSTANTS, INVENTORY_CONSTANTS, RESOURCE_LIMITS
)
from sierra.environment.entities import Resource, Agent
from sierra.environment.grid import place_entity, get_cell_content, create_grid

# Base step penalty
STEP_PENALTY = -0.01

# Expected rewards for collection (resource_value from _handle_movement_and_collection)
COLLECTION_REWARDS = {
    'food': 0.5,
    'water': 0.5,
    'wood': 0.1,
    'stone': 0.1,
    'charcoal': 0.1,
    'cloth': 0.1,
    'murky_water': 0.1
}

class TestResourceCollection:

    def setup_method(self):
        self.env = SierraEnv(grid_width=10, grid_height=10) # Use a slightly larger grid
        self.env.reset()
        # Ensure agent starts at a known, controllable position for easier testing
        self.env.agent.x = 1
        self.env.agent.y = 1
        # Clear grid and resources, then place agent
        self.env.grid = create_grid(self.env.grid_width, self.env.grid_height)
        self.env.resources = []
        place_entity(self.env.grid, self.env.agent, AGENT, x=self.env.agent.x, y=self.env.agent.y)


    def place_resource_near_agent(self, resource_type, agent_pos_override=None, offset_x=1, offset_y=0):
        agent_x = agent_pos_override[0] if agent_pos_override else self.env.agent.x
        agent_y = agent_pos_override[1] if agent_pos_override else self.env.agent.y
        
        res_x, res_y = agent_x + offset_x, agent_y + offset_y

        if not (0 <= res_x < self.env.grid_width and 0 <= res_y < self.env.grid_height):
            return False # Out of bounds

        # Clear any existing resource at target from self.env.resources
        self.env.resources = [r for r in self.env.resources if not (r.x == res_x and r.y == res_y)]
        # Ensure grid cell is initially empty or becomes empty for the new resource
        self.env.grid[res_y, res_x] = EMPTY 
        
        new_res = Resource(res_x, res_y, type=resource_type)
        self.env.resources.append(new_res)
        place_entity(self.env.grid, new_res, RESOURCE, x=res_x, y=res_y)
        return True

    def move_and_collect(self, move_action):
        return self.env.step(move_action)

    # 1. Test Collect Each Resource Type (No Axe)
    RESOURCE_TYPES_TO_TEST = ['food', 'water', 'wood', 'stone', 'charcoal', 'cloth', 'murky_water']

    @pytest.mark.parametrize("resource_type", RESOURCE_TYPES_TO_TEST)
    def test_collect_each_resource_type_no_axe(self, resource_type):
        self.env.agent.has_axe = False
        self.env.agent.inventory[resource_type] = 0
        
        initial_agent_x, initial_agent_y = self.env.agent.x, self.env.agent.y
        resource_x, resource_y = initial_agent_x + 1, initial_agent_y
        
        assert self.place_resource_near_agent(resource_type, agent_pos_override=(initial_agent_x, initial_agent_y)), f"Failed to place {resource_type}"

        obs, reward, _, _, _ = self.move_and_collect(Actions.MOVE_RIGHT.value)

        assert self.env.agent.inventory[resource_type] == 1, f"Inventory check failed for {resource_type}"
        expected_reward = COLLECTION_REWARDS[resource_type] + STEP_PENALTY
        assert reward == pytest.approx(expected_reward), f"Reward check failed for {resource_type}"
        
        assert not any(r.x == resource_x and r.y == resource_y for r in self.env.resources), f"Resource not removed for {resource_type}"
        # Agent moves onto the cell, so it should contain AGENT. The resource itself is gone.
        assert get_cell_content(self.env.grid, resource_x, resource_y) == AGENT, f"Grid content check failed for {resource_type}"
        assert get_cell_content(self.env.grid, initial_agent_x, initial_agent_y) == EMPTY, f"Old agent cell not empty for {resource_type}"
        
        assert self.env.agent.x == resource_x and self.env.agent.y == resource_y, f"Agent position update failed for {resource_type}"
        assert np.array_equal(obs[f"{resource_type}_locs" if resource_type != "murky_water" else "murky_water_locs"], 
                              np.array([[-1,-1]] * RESOURCE_LIMITS[f"MAX_{resource_type.upper()}_SOURCES"], dtype=int)), f"Observation locs failed for {resource_type}"
        
        # For food and water, inventory is not directly observed in 'inv_...' keys. Their effect is on hunger/thirst.
        # For materials, their inventory is observed.
        if resource_type not in ['food', 'water']:
            assert obs[f"inv_{resource_type}" if resource_type != "murky_water" else "inv_murky_water"][0] == 1, f"Observation inventory failed for {resource_type}"
        
        assert np.array_equal(obs["agent_pos"], np.array([resource_x, resource_y])), f"Observation agent_pos failed for {resource_type}"

    # 2. Test Collect Wood (With Axe)
    def test_collect_wood_with_axe(self):
        self.env.agent.has_axe = True
        self.env.agent.inventory["wood"] = 0
        initial_agent_x, initial_agent_y = self.env.agent.x, self.env.agent.y
        resource_x, resource_y = initial_agent_x + 1, initial_agent_y

        assert self.place_resource_near_agent("wood", agent_pos_override=(initial_agent_x, initial_agent_y)), "Failed to place wood"

        obs, reward, _, _, _ = self.move_and_collect(Actions.MOVE_RIGHT.value)

        assert self.env.agent.inventory["wood"] == GAMEPLAY_CONSTANTS['WOOD_COLLECTION_AXE_BONUS']
        expected_reward = COLLECTION_REWARDS['wood'] + STEP_PENALTY
        assert reward == pytest.approx(expected_reward)
        assert not any(r.x == resource_x and r.y == resource_y for r in self.env.resources)
        assert get_cell_content(self.env.grid, resource_x, resource_y) == AGENT
        assert self.env.agent.x == resource_x and self.env.agent.y == resource_y
        assert obs["inv_wood"][0] == GAMEPLAY_CONSTANTS['WOOD_COLLECTION_AXE_BONUS']

    # 3. Test Collection up to Inventory Limit
    def test_collect_wood_inventory_limit(self):
        self.env.agent.has_axe = False # Collect 1 at a time
        self.env.agent.inventory["wood"] = INVENTORY_CONSTANTS['MAX_INVENTORY_PER_ITEM'] - 1
        
        # First collection to reach limit
        agent_current_x, agent_current_y = self.env.agent.x, self.env.agent.y
        resource_x1, resource_y1 = agent_current_x + 1, agent_current_y
        assert self.place_resource_near_agent("wood", agent_pos_override=(agent_current_x, agent_current_y)), "Failed to place wood (1)"
        
        self.move_and_collect(Actions.MOVE_RIGHT.value)
        assert self.env.agent.inventory["wood"] == INVENTORY_CONSTANTS['MAX_INVENTORY_PER_ITEM']
        
        # Agent is now at (resource_x1, resource_y1). Place another wood to its right.
        agent_current_x, agent_current_y = self.env.agent.x, self.env.agent.y
        resource_x2, resource_y2 = agent_current_x + 1, agent_current_y

        # Ensure agent is not at the edge before placing next resource
        if agent_current_x >= self.env.grid_width - 1:
            pytest.skip("Agent at edge, cannot place second resource for limit test")

        assert self.place_resource_near_agent("wood", agent_pos_override=(agent_current_x, agent_current_y)), "Failed to place wood (2)"
        
        obs, reward, _, _, _ = self.move_and_collect(Actions.MOVE_RIGHT.value)
        assert self.env.agent.inventory["wood"] == INVENTORY_CONSTANTS['MAX_INVENTORY_PER_ITEM'] # Should still be maxed
        # Reward should still include collection reward, as item is collected but inventory doesn't change due to cap
        expected_reward_second_collect = COLLECTION_REWARDS['wood'] + STEP_PENALTY 
        assert reward == pytest.approx(expected_reward_second_collect)


    # 4. Test Collection of Non-Existent Resource (Empty Cell)
    def test_move_to_empty_cell(self):
        initial_agent_x, initial_agent_y = self.env.agent.x, self.env.agent.y
        target_x, target_y = initial_agent_x + 1, initial_agent_y
        
        # Ensure cell is empty
        self.env.grid[target_y, target_x] = EMPTY
        self.env.resources = [r for r in self.env.resources if not (r.x == target_x and r.y == target_y)]
        
        initial_inventory = self.env.agent.inventory.copy()

        obs, reward, _, _, _ = self.move_and_collect(Actions.MOVE_RIGHT.value)

        assert self.env.agent.inventory == initial_inventory
        assert reward == pytest.approx(STEP_PENALTY) # Only step penalty
        assert self.env.agent.x == target_x and self.env.agent.y == target_y
        assert get_cell_content(self.env.grid, target_x, target_y) == AGENT
        assert get_cell_content(self.env.grid, initial_agent_x, initial_agent_y) == EMPTY

    # 5. Test Resource List Correctly Updated
    def test_resource_list_updates_after_collection(self):
        agent_start_x, agent_start_y = self.env.agent.x, self.env.agent.y
        
        # Place food to the right
        food_x, food_y = agent_start_x + 1, agent_start_y
        assert self.place_resource_near_agent("food", agent_pos_override=(agent_start_x, agent_start_y), offset_x=1, offset_y=0)
        food_resource_obj = next(r for r in self.env.resources if r.x == food_x and r.y == food_y and r.type == "food")

        # Place wood two steps to the right (or any other distinct location)
        wood_x, wood_y = agent_start_x + 2, agent_start_y
        if wood_x >= self.env.grid_width: # Ensure wood placement is valid
             pytest.skip("Grid too small to place second distinct resource for this test logic")
        assert self.place_resource_near_agent("wood", agent_pos_override=(agent_start_x, agent_start_y), offset_x=2, offset_y=0)
        wood_resource_obj = next(r for r in self.env.resources if r.x == wood_x and r.y == wood_y and r.type == "wood")
        
        initial_resource_count = len(self.env.resources)
        assert initial_resource_count == 2

        # Collect food
        self.move_and_collect(Actions.MOVE_RIGHT.value) # Agent moves to (food_x, food_y)

        assert food_resource_obj not in self.env.resources # Check specific object
        assert not any(r.x == food_x and r.y == food_y for r in self.env.resources)
        assert wood_resource_obj in self.env.resources # Wood should still be there
        assert any(r.x == wood_x and r.y == wood_y and r.type == "wood" for r in self.env.resources)
        assert len(self.env.resources) == initial_resource_count - 1
