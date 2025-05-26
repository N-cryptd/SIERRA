import pytest
import numpy as np

from sierra.environment.core import (
    SierraEnv, Actions, GAMEPLAY_CONSTANTS, CRAFTING_RECIPES, INVENTORY_CONSTANTS, TIME_CONSTANTS
)
from sierra.environment.entities import Agent # Resource not strictly needed for these tests

# Constants from _update_agent_needs in core.py
BASE_DECAY = 0.1
SHELTER_MULTIPLIER = 0.75
NIGHT_MULTIPLIER = 1.2
NIGHT_NO_SHELTER_EXTRA_MULTIPLIER = 1.5 # Applied on top of NIGHT_MULTIPLIER

# Reward/Penalty Constants
STEP_PENALTY = -0.01
DEATH_PENALTY = -1.0
LOW_NEED_PENALTY = 0.05 # From core.py logic: reward -= 0.05 if hunger/thirst <= LOW_NEED_THRESHOLD (20)

class TestAgentNeeds:

    def setup_method(self):
        self.env = SierraEnv()
        self.env.reset()
        # Standardize agent state for most tests
        self.env.agent.hunger = 100.0
        self.env.agent.thirst = 100.0
        self.env.agent.has_shelter = False
        self.env.agent.has_axe = False # Not relevant here but good for consistency
        self.env.agent.inventory = {mat: 0 for mat in self.env.agent.inventory.keys()} # Clear inventory
        self.env.is_day = True # Default to day

    def _perform_n_no_op_steps(self, n_steps):
        """Performs N steps of a no-op action (like failed crafting)."""
        # Crafting shelter when already owned (if owned) or without materials (if not owned)
        # is a reliable no-op that doesn't change inventory or position.
        action_to_use = Actions.CRAFT_SHELTER.value
        
        last_obs = None
        total_reward = 0
        terminated = False
        truncated = False

        for _ in range(n_steps):
            if terminated: # Stop if environment terminates early
                break
            # Ensure agent has shelter for a reliable no-op if that's the chosen action's side effect
            # For these tests, we primarily care about _update_agent_needs, so a failing craft is fine.
            # If agent has no shelter, and no mats, CRAFT_SHELTER fails.
            # If agent has shelter, CRAFT_SHELTER fails.
            obs, reward, term, trunc, _ = self.env.step(action_to_use)
            last_obs = obs
            total_reward += reward
            terminated = term
            truncated = trunc
        return last_obs, total_reward, terminated, truncated

    # 1. Test Basic Needs Decay (Day, No Shelter)
    def test_needs_decay_day_no_shelter(self):
        self.env.is_day = True
        self.env.agent.has_shelter = False
        n_steps = 3
        
        expected_hunger_decay_per_step = BASE_DECAY
        expected_thirst_decay_per_step = BASE_DECAY

        obs, _, _, _ = self._perform_n_no_op_steps(n_steps)

        expected_hunger = 100.0 - (expected_hunger_decay_per_step * n_steps)
        expected_thirst = 100.0 - (expected_thirst_decay_per_step * n_steps)

        assert self.env.agent.hunger == pytest.approx(expected_hunger)
        assert self.env.agent.thirst == pytest.approx(expected_thirst)
        assert obs["hunger"][0] == pytest.approx(expected_hunger)
        assert obs["thirst"][0] == pytest.approx(expected_thirst)

    # 2. Test Needs Decay (Night, No Shelter)
    def test_needs_decay_night_no_shelter(self):
        self.env.is_day = False
        self.env.agent.has_shelter = False
        # Set world_time to ensure it's night and won't flip to day in 1 step
        self.env.world_time = TIME_CONSTANTS["DAY_LENGTH"] 
        self.env.is_day = (self.env.world_time % (TIME_CONSTANTS["DAY_LENGTH"] + TIME_CONSTANTS["NIGHT_LENGTH"])) < TIME_CONSTANTS["DAY_LENGTH"]
        assert not self.env.is_day # Should be night

        n_steps = 1 # Test for 1 step to avoid is_day flipping and simplify calculation

        expected_hunger_decay_per_step = BASE_DECAY * NIGHT_MULTIPLIER * NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
        expected_thirst_decay_per_step = BASE_DECAY * NIGHT_MULTIPLIER * NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
        
        obs, _, _, _ = self._perform_n_no_op_steps(n_steps)

        expected_hunger = 100.0 - expected_hunger_decay_per_step # Only one step
        expected_thirst = 100.0 - expected_thirst_decay_per_step # Only one step

        assert self.env.agent.hunger == pytest.approx(expected_hunger)
        assert self.env.agent.thirst == pytest.approx(expected_thirst)
        assert obs["hunger"][0] == pytest.approx(expected_hunger)
        assert obs["thirst"][0] == pytest.approx(expected_thirst)

    # 3. Test Needs Decay (Day, With Shelter)
    def test_needs_decay_day_with_shelter(self):
        self.env.is_day = True
        self.env.agent.has_shelter = True # Agent has shelter
        n_steps = 3

        expected_hunger_decay_per_step = BASE_DECAY * SHELTER_MULTIPLIER
        expected_thirst_decay_per_step = BASE_DECAY * SHELTER_MULTIPLIER

        obs, _, _, _ = self._perform_n_no_op_steps(n_steps)

        expected_hunger = 100.0 - (expected_hunger_decay_per_step * n_steps)
        expected_thirst = 100.0 - (expected_thirst_decay_per_step * n_steps)
        
        assert self.env.agent.hunger == pytest.approx(expected_hunger)
        assert self.env.agent.thirst == pytest.approx(expected_thirst)
        assert obs["hunger"][0] == pytest.approx(expected_hunger)
        assert obs["thirst"][0] == pytest.approx(expected_thirst)

    # 4. Test Needs Decay (Night, With Shelter)
    def test_needs_decay_night_with_shelter(self):
        self.env.is_day = False
        self.env.agent.has_shelter = True # Agent has shelter
        self.env.world_time = TIME_CONSTANTS["DAY_LENGTH"]
        self.env.is_day = (self.env.world_time % (TIME_CONSTANTS["DAY_LENGTH"] + TIME_CONSTANTS["NIGHT_LENGTH"])) < TIME_CONSTANTS["DAY_LENGTH"]
        assert not self.env.is_day

        n_steps = 1 # Test for 1 step

        expected_hunger_decay_per_step = BASE_DECAY * SHELTER_MULTIPLIER * NIGHT_MULTIPLIER
        expected_thirst_decay_per_step = BASE_DECAY * SHELTER_MULTIPLIER * NIGHT_MULTIPLIER
        
        obs, _, _, _ = self._perform_n_no_op_steps(n_steps)

        expected_hunger = 100.0 - expected_hunger_decay_per_step
        expected_thirst = 100.0 - expected_thirst_decay_per_step

        assert self.env.agent.hunger == pytest.approx(expected_hunger)
        assert self.env.agent.thirst == pytest.approx(expected_thirst)
        assert obs["hunger"][0] == pytest.approx(expected_hunger)
        assert obs["thirst"][0] == pytest.approx(expected_thirst)

    # 5. Test Thirst Replenishment (Purify Water)
    def test_thirst_replenishment_purify_water(self):
        self.env.agent.water_filters_available = 1
        self.env.agent.inventory["murky_water"] = 1
        self.env.agent.thirst = 50.0
        self.env.agent.hunger = 50.0 # Set hunger too to check its decay
        self.env.is_day = True # Ensure known decay rate for hunger
        self.env.agent.has_shelter = False

        initial_thirst = self.env.agent.thirst
        initial_hunger = self.env.agent.hunger

        obs, reward, _, _, info = self.env.step(Actions.PURIFY_WATER.value) # Corrected unpacking
        
        # Thirst: replenished then decays
        expected_thirst_after_replenish = min(100, initial_thirst + GAMEPLAY_CONSTANTS['PURIFIED_WATER_THIRST_REPLENISH'])
        expected_thirst_decay_this_step = BASE_DECAY # Day, no shelter
        final_expected_thirst = expected_thirst_after_replenish - expected_thirst_decay_this_step
        
        # Hunger: just decays
        expected_hunger_decay_this_step = BASE_DECAY # Day, no shelter
        final_expected_hunger = initial_hunger - expected_hunger_decay_this_step

        assert self.env.agent.thirst == pytest.approx(final_expected_thirst)
        assert self.env.agent.hunger == pytest.approx(final_expected_hunger)
        assert obs["thirst"][0] == pytest.approx(final_expected_thirst)
        assert obs["hunger"][0] == pytest.approx(final_expected_hunger)
        assert reward == pytest.approx(0.3 + STEP_PENALTY) # 0.3 for purify, -0.01 step penalty

    # 6. Test Needs Do Not Go Below Zero
    def test_needs_do_not_go_below_zero(self):
        self.env.agent.hunger = 0.05
        self.env.agent.thirst = 0.03
        self.env.is_day = True # Base decay of 0.1
        self.env.agent.has_shelter = False

        obs, _, _, _ = self._perform_n_no_op_steps(1) # One step is enough

        assert self.env.agent.hunger == 0.0
        assert self.env.agent.thirst == 0.0
        assert obs["hunger"][0] == 0.0
        assert obs["thirst"][0] == 0.0

    # 7. Test Termination on Zero Hunger
    def test_termination_on_zero_hunger(self):
        self.env.agent.hunger = 0.05 # Will become 0 after decay
        self.env.agent.thirst = 50.0
        self.env.is_day = True
        self.env.agent.has_shelter = False

        obs, reward, terminated, _ = self._perform_n_no_op_steps(1)
        
        assert self.env.agent.hunger == 0.0
        assert terminated is True
        assert reward == pytest.approx(DEATH_PENALTY + STEP_PENALTY - LOW_NEED_PENALTY)
        assert obs["hunger"][0] == 0.0

    # 8. Test Termination on Zero Thirst
    def test_termination_on_zero_thirst(self):
        self.env.agent.hunger = 50.0
        self.env.agent.thirst = 0.05 # Will become 0 after decay
        self.env.is_day = True
        self.env.agent.has_shelter = False

        obs, reward, terminated, _ = self._perform_n_no_op_steps(1)

        assert self.env.agent.thirst == 0.0
        assert terminated is True
        assert reward == pytest.approx(DEATH_PENALTY + STEP_PENALTY - LOW_NEED_PENALTY)
        assert obs["thirst"][0] == 0.0

    # 9. Test Needs After Food/Water Collection (Skipped)
    @pytest.mark.skip(reason="Direct consumption of food/water from inventory is not yet implemented")
    def test_direct_consumption_replenishment(self):
        pass
