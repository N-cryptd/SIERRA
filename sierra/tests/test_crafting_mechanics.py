import pytest
import numpy as np
from sierra.environment.core import (
    SierraEnv, ACTION_CRAFT_SHELTER, ACTION_CRAFT_FILTER, ACTION_CRAFT_AXE,
    ACTION_PURIFY_WATER, WOOD_COLLECTION_AXE_BONUS, PURIFIED_WATER_THIRST_REPLENISH,
    MAX_INVENTORY_PER_ITEM, MAX_WATER_FILTERS, CRAFTING_REWARDS,
    RESOURCE, EMPTY, AGENT, ACTION_MOVE_RIGHT # Assuming ACTION_MOVE_RIGHT is 3
)
from sierra.environment.entities import Resource, Agent

# Helper to set agent inventory easily
def set_inventory(agent, items):
    for item, count in items.items():
        agent.inventory[item] = count

class TestCraftingMechanics:

    def setup_method(self):
        self.env = SierraEnv()
        self.env.reset() # Important to initialize agent and resources

    # Helper for placing a resource (can be more sophisticated)
    def place_resource_near_agent(self, resource_type, agent_pos, offset_x=1, offset_y=0):
        # For simplicity, place it one step to the right if possible
        res_x, res_y = agent_pos[0] + offset_x, agent_pos[1] + offset_y
        
        # Ensure coordinates are within grid boundaries
        if not (0 <= res_x < self.env.grid_width and 0 <= res_y < self.env.grid_height):
            # print(f"Cannot place resource: ({res_x},{res_y}) is outside grid ({self.env.grid_width}x{self.env.grid_height})")
            return False

        # Clear cell first if occupied by another resource for test predictability
        # For test simplicity, assume it's empty or handle clearing
        # Also ensure it's not placing on the agent itself if offset is 0,0
        if self.env.grid[res_y, res_x] == AGENT and not (offset_x==0 and offset_y==0) : # Allow placing on agent if offset is 0,0 for some tests
             # print(f"Cannot place resource: ({res_x},{res_y}) is occupied by AGENT")
             return False # Cannot place on agent unless specifically intended (which this helper isn't for)

        self.env.grid[res_y, res_x] = RESOURCE
        new_res = Resource(res_x, res_y, type=resource_type)
        
        # Remove any existing resource at that spot from self.env.resources
        self.env.resources = [r for r in self.env.resources if not (r.x == res_x and r.y == res_y)]
        self.env.resources.append(new_res)
        # print(f"Placed {resource_type} at ({res_x},{res_y}). Resources: {[(r.type, r.x, r.y) for r in self.env.resources]}")
        return True

    # -------------------------------------------
    # 1. Test Successful Crafting
    # -------------------------------------------
    def test_craft_shelter_success(self):
        set_inventory(self.env.agent, {"wood": 4, "stone": 2})
        initial_wood = self.env.agent.inventory["wood"]
        initial_stone = self.env.agent.inventory["stone"]

        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_SHELTER)

        assert self.env.agent.inventory["wood"] == initial_wood - 4
        assert self.env.agent.inventory["stone"] == initial_stone - 2
        assert self.env.agent.has_shelter is True
        assert obs["has_shelter"] == 1
        assert reward == CRAFTING_REWARDS["basic_shelter"] - 0.01 # Step penalty

    def test_craft_filter_success(self):
        set_inventory(self.env.agent, {"charcoal": 2, "cloth": 1, "stone": 1})
        initial_charcoal = self.env.agent.inventory["charcoal"]
        initial_cloth = self.env.agent.inventory["cloth"]
        initial_stone = self.env.agent.inventory["stone"]

        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_FILTER)

        assert self.env.agent.inventory["charcoal"] == initial_charcoal - 2
        assert self.env.agent.inventory["cloth"] == initial_cloth - 1
        assert self.env.agent.inventory["stone"] == initial_stone - 1
        assert self.env.agent.water_filters_available == 1
        assert obs["water_filters_available"][0] == 1
        assert reward == CRAFTING_REWARDS["water_filter"] - 0.01 # Step penalty

    def test_craft_axe_success(self):
        set_inventory(self.env.agent, {"stone": 2, "wood": 1})
        initial_stone = self.env.agent.inventory["stone"]
        initial_wood = self.env.agent.inventory["wood"]

        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_AXE)

        assert self.env.agent.inventory["stone"] == initial_stone - 2
        assert self.env.agent.inventory["wood"] == initial_wood - 1
        assert self.env.agent.has_axe is True
        assert obs["has_axe"] == 1
        assert reward == CRAFTING_REWARDS["crude_axe"] - 0.01 # Step penalty

    # -------------------------------------------
    # 2. Test Crafting Failure - Insufficient Materials
    # -------------------------------------------
    def test_craft_shelter_fail_no_materials(self):
        set_inventory(self.env.agent, {"wood": 1, "stone": 1}) # Insufficient
        initial_inventory = self.env.agent.inventory.copy()

        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_SHELTER)

        assert self.env.agent.inventory["wood"] == initial_inventory["wood"]
        assert self.env.agent.inventory["stone"] == initial_inventory["stone"]
        assert self.env.agent.has_shelter is False
        assert obs["has_shelter"] == 0
        assert reward == -0.01 # Only step penalty

    def test_craft_filter_fail_no_materials(self):
        set_inventory(self.env.agent, {"charcoal": 1, "cloth": 0, "stone": 1}) # Insufficient
        initial_inventory = self.env.agent.inventory.copy()

        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_FILTER)

        assert self.env.agent.inventory["charcoal"] == initial_inventory["charcoal"]
        assert self.env.agent.inventory["cloth"] == initial_inventory["cloth"]
        assert self.env.agent.inventory["stone"] == initial_inventory["stone"]
        assert self.env.agent.water_filters_available == 0
        assert obs["water_filters_available"][0] == 0
        assert reward == -0.01 # Only step penalty

    def test_craft_axe_fail_no_materials(self):
        set_inventory(self.env.agent, {"stone": 1, "wood": 0}) # Insufficient
        initial_inventory = self.env.agent.inventory.copy()

        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_AXE)

        assert self.env.agent.inventory["stone"] == initial_inventory["stone"]
        assert self.env.agent.inventory["wood"] == initial_inventory["wood"]
        assert self.env.agent.has_axe is False
        assert obs["has_axe"] == 0
        assert reward == -0.01 # Only step penalty

    # -------------------------------------------
    # 3. Test Crafting Failure - Item Already Owned/Maxed
    # -------------------------------------------
    def test_craft_shelter_fail_already_owned(self):
        set_inventory(self.env.agent, {"wood": 8, "stone": 4}) # Enough for two
        self.env.step(ACTION_CRAFT_SHELTER) # First successful craft
        assert self.env.agent.has_shelter is True
        
        initial_inventory_after_first_craft = self.env.agent.inventory.copy()
        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_SHELTER) # Attempt second craft

        assert self.env.agent.inventory == initial_inventory_after_first_craft # No change
        assert self.env.agent.has_shelter is True # Still true
        assert obs["has_shelter"] == 1
        assert reward == -0.01 # Only step penalty for second attempt

    def test_craft_axe_fail_already_owned(self):
        set_inventory(self.env.agent, {"stone": 4, "wood": 2}) # Enough for two
        self.env.step(ACTION_CRAFT_AXE) # First successful craft
        assert self.env.agent.has_axe is True

        initial_inventory_after_first_craft = self.env.agent.inventory.copy()
        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_AXE) # Attempt second craft

        assert self.env.agent.inventory == initial_inventory_after_first_craft
        assert self.env.agent.has_axe is True
        assert obs["has_axe"] == 1
        assert reward == -0.01 # Only step penalty

    def test_craft_filter_fail_maxed(self):
        set_inventory(self.env.agent, {"charcoal": 2 * MAX_WATER_FILTERS + 2, 
                                     "cloth": 1 * MAX_WATER_FILTERS + 1, 
                                     "stone": 1 * MAX_WATER_FILTERS + 1})
        for _ in range(MAX_WATER_FILTERS):
            _, _, _, _, _ = self.env.step(ACTION_CRAFT_FILTER)
        
        assert self.env.agent.water_filters_available == MAX_WATER_FILTERS
        
        initial_inventory_after_max_crafts = self.env.agent.inventory.copy()
        obs, reward, _, _, _ = self.env.step(ACTION_CRAFT_FILTER) # Attempt to craft one more

        assert self.env.agent.inventory == initial_inventory_after_max_crafts
        assert self.env.agent.water_filters_available == MAX_WATER_FILTERS
        assert obs["water_filters_available"][0] == MAX_WATER_FILTERS
        assert reward == -0.01 # Only step penalty

    # -------------------------------------------
    # 4. Test Item Effects
    # -------------------------------------------
    def test_shelter_effect_decay(self):
        # With shelter
        self.env.reset()
        self.env.agent.has_shelter = True
        initial_hunger = self.env.agent.hunger
        initial_thirst = self.env.agent.thirst
        
        num_steps = 3
        expected_hunger_decay_per_step = 0.1 * 0.75 # Assuming day time
        expected_thirst_decay_per_step = 0.1 * 0.75 # Assuming day time
        if not self.env.is_day: # Should be day after reset, but for robustness
            expected_hunger_decay_per_step *= 1.2
            expected_thirst_decay_per_step *= 1.2

        for i in range(num_steps):
            # Use an action that doesn't affect hunger/thirst directly, e.g., move into wall
            # To ensure agent doesn't move, find a wall. Or just use a non-move action if available.
            # For simplicity, let's assume agent is at (0,0) and try to move up (action 0)
            # This will hit boundary if at edge, or move if not.
            # A better way: ensure agent doesn't collect anything.
            # Crafting actions (that fail) are good as they don't cause movement.
            self.env.step(ACTION_CRAFT_SHELTER) # Agent already has shelter, so this will fail & not move

        expected_hunger = initial_hunger - (expected_hunger_decay_per_step * num_steps)
        expected_thirst = initial_thirst - (expected_thirst_decay_per_step * num_steps)
        
        assert abs(self.env.agent.hunger - expected_hunger) < 1e-5
        assert abs(self.env.agent.thirst - expected_thirst) < 1e-5

        # Without shelter
        self.env.reset()
        self.env.agent.has_shelter = False
        initial_hunger_no_shelter = self.env.agent.hunger
        initial_thirst_no_shelter = self.env.agent.thirst

        expected_hunger_decay_no_shelter = 0.1 # Assuming day time
        expected_thirst_decay_no_shelter = 0.1 # Assuming day time
        if not self.env.is_day:
            expected_hunger_decay_no_shelter *= 1.2
            expected_thirst_decay_no_shelter *= 1.2

        for i in range(num_steps):
            self.env.step(ACTION_CRAFT_SHELTER) # Agent has no shelter, and likely no mats, so fails & no move

        expected_hunger_ns = initial_hunger_no_shelter - (expected_hunger_decay_no_shelter * num_steps)
        expected_thirst_ns = initial_thirst_no_shelter - (expected_thirst_decay_no_shelter * num_steps)

        assert abs(self.env.agent.hunger - expected_hunger_ns) < 1e-5
        assert abs(self.env.agent.thirst - expected_thirst_ns) < 1e-5


    def test_axe_effect_collection(self):
        # Without axe
        self.env.reset()
        self.env.agent.inventory["wood"] = 0
        agent_start_pos = (self.env.agent.x, self.env.agent.y)
        # Ensure agent is not at the rightmost edge
        if agent_start_pos[0] == self.env.grid_width -1:
            self.env.agent.x -=1 # move agent left if at edge
            agent_start_pos = (self.env.agent.x, self.env.agent.y)
            self.env.grid[agent_start_pos[1], agent_start_pos[0]] = AGENT
            self.env.grid[agent_start_pos[1], agent_start_pos[0]+1] = EMPTY


        assert self.place_resource_near_agent("wood", agent_start_pos), "Failed to place wood for test"
        
        self.env.step(ACTION_MOVE_RIGHT) # Collect wood
        assert self.env.agent.inventory["wood"] == 1

        # With axe
        self.env.reset()
        self.env.agent.has_axe = True
        self.env.agent.inventory["wood"] = 0
        agent_start_pos_axe = (self.env.agent.x, self.env.agent.y)
        if agent_start_pos_axe[0] == self.env.grid_width -1:
            self.env.agent.x -=1
            agent_start_pos_axe = (self.env.agent.x, self.env.agent.y)
            self.env.grid[agent_start_pos_axe[1], agent_start_pos_axe[0]] = AGENT
            self.env.grid[agent_start_pos_axe[1], agent_start_pos_axe[0]+1] = EMPTY


        assert self.place_resource_near_agent("wood", agent_start_pos_axe), "Failed to place wood for axe test"
        
        self.env.step(ACTION_MOVE_RIGHT) # Collect wood
        assert self.env.agent.inventory["wood"] == WOOD_COLLECTION_AXE_BONUS


    def test_water_filter_purify_action_success(self):
        self.env.reset()
        self.env.agent.water_filters_available = 1
        set_inventory(self.env.agent, {"murky_water": 1})
        self.env.agent.thirst = 50 # Set thirst to a value that won't cap immediately
        
        initial_thirst = self.env.agent.thirst
        obs, reward, _, _, _ = self.env.step(ACTION_PURIFY_WATER)

        assert self.env.agent.water_filters_available == 0
        assert self.env.agent.inventory["murky_water"] == 0
        expected_thirst = min(100, initial_thirst + PURIFIED_WATER_THIRST_REPLENISH)
        # Account for the 0.1 thirst decay that happens in the step
        assert self.env.agent.thirst == pytest.approx(expected_thirst - 0.1, abs=1e-5)
        assert obs["water_filters_available"][0] == 0
        assert obs["inventory"]["murky_water"][0] == 0
        assert obs["thirst"][0] == pytest.approx(expected_thirst - 0.1, abs=1e-5)
        assert reward == 0.3 - 0.01 # Purify reward - step penalty

    def test_water_filter_purify_action_fail_no_filter(self):
        self.env.reset()
        self.env.agent.water_filters_available = 0 # No filter
        set_inventory(self.env.agent, {"murky_water": 1})
        initial_thirst = self.env.agent.thirst
        initial_murky_water = self.env.agent.inventory["murky_water"]

        obs, reward, _, _, _ = self.env.step(ACTION_PURIFY_WATER)

        assert self.env.agent.water_filters_available == 0
        assert self.env.agent.inventory["murky_water"] == initial_murky_water
        # Account for the 0.1 thirst decay
        assert self.env.agent.thirst == pytest.approx(initial_thirst - 0.1, abs=1e-5)
        assert reward == -0.01 # Step penalty only

    def test_water_filter_purify_action_fail_no_murky_water(self):
        self.env.reset()
        self.env.agent.water_filters_available = 1
        set_inventory(self.env.agent, {"murky_water": 0}) # No murky water
        initial_thirst = self.env.agent.thirst
        initial_filters = self.env.agent.water_filters_available

        obs, reward, _, _, _ = self.env.step(ACTION_PURIFY_WATER)

        assert self.env.agent.water_filters_available == initial_filters # No change
        assert self.env.agent.inventory["murky_water"] == 0
        # Account for the 0.1 thirst decay
        assert self.env.agent.thirst == pytest.approx(initial_thirst - 0.1, abs=1e-5)
        assert reward == -0.01 # Step penalty only

    # -------------------------------------------
    # 5. Test Inventory Limits
    # -------------------------------------------
    def test_inventory_limits_wood(self):
        self.env.reset()
        self.env.agent.inventory["wood"] = MAX_INVENTORY_PER_ITEM - 1
        
        agent_pos = (self.env.agent.x, self.env.agent.y)
        if agent_pos[0] == self.env.grid_width -1: self.env.agent.x -=1; agent_pos = (self.env.agent.x, self.env.agent.y)
        self.env.grid[agent_pos[1], agent_pos[0]] = AGENT # ensure agent is there
        self.env.grid[agent_pos[1], agent_pos[0]+1] = EMPTY # ensure next cell is empty for resource

        assert self.place_resource_near_agent("wood", agent_pos), "Failed to place wood (1)"
        self.env.step(ACTION_MOVE_RIGHT) # Collect first wood
        assert self.env.agent.inventory["wood"] == MAX_INVENTORY_PER_ITEM

        # Agent is now at the position of the first resource. Place another one to its right.
        current_agent_pos = (self.env.agent.x, self.env.agent.y)
        if current_agent_pos[0] == self.env.grid_width -1: # If agent moved to edge
             pytest.skip("Agent at edge, cannot place another resource to the right for this test logic")

        assert self.place_resource_near_agent("wood", current_agent_pos), "Failed to place wood (2)"
        self.env.step(ACTION_MOVE_RIGHT) # Attempt to collect second wood
        assert self.env.agent.inventory["wood"] == MAX_INVENTORY_PER_ITEM # Still maxed

    # -------------------------------------------
    # 6. Test Murky Water Collection
    # -------------------------------------------
    def test_murky_water_collection(self):
        self.env.reset()
        self.env.agent.inventory["murky_water"] = 0
        
        agent_pos = (self.env.agent.x, self.env.agent.y)
        if agent_pos[0] == self.env.grid_width -1: self.env.agent.x -=1; agent_pos = (self.env.agent.x, self.env.agent.y)
        self.env.grid[agent_pos[1], agent_pos[0]] = AGENT
        self.env.grid[agent_pos[1], agent_pos[0]+1] = EMPTY


        assert self.place_resource_near_agent("murky_water", agent_pos), "Failed to place murky_water"
        
        obs, reward, _, _, _ = self.env.step(ACTION_MOVE_RIGHT) # Collect murky water
        
        assert self.env.agent.inventory["murky_water"] == 1
        assert obs["inventory"]["murky_water"][0] == 1
        # Expected reward for murky water collection is 0.1, plus step penalty -0.01
        assert abs(reward - (0.1 - 0.01)) < 1e-5
