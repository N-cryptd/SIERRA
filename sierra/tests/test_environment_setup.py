import pytest
import numpy as np
from collections import Counter

from sierra.environment.core import SierraEnv, RESOURCE_LIMITS, AGENT, RESOURCE, EMPTY
from sierra.environment.entities import Agent, Resource
from sierra.environment.grid import get_cell_content

class TestEnvironmentSetup:

    def setup_method(self):
        self.env = SierraEnv(grid_width=10, grid_height=10) # Standard size for testing setup

    def test_agent_placed_on_reset(self):
        """Verify agent is placed correctly on reset."""
        obs, info = self.env.reset()
        assert self.env.agent is not None, "Agent object not created."
        agent_x, agent_y = self.env.agent.x, self.env.agent.y
        assert get_cell_content(self.env.grid, agent_x, agent_y) == AGENT, \
            f"Agent not found at its coordinates ({agent_x},{agent_y}) on the grid. Cell content: {self.env.grid[agent_y, agent_x]}"
        assert np.array_equal(obs["agent_pos"], np.array([agent_x, agent_y])), \
            "Agent position in observation does not match agent's actual position."

    def test_correct_number_of_resources_spawned(self):
        """Verify the correct number of each resource type is spawned."""
        self.env.reset()
        
        spawned_resource_counts = Counter(r.type for r in self.env.resources)
        
        expected_total_resources = 0
        for resource_type, expected_count in RESOURCE_LIMITS.items():
            # RESOURCE_LIMITS keys are like "MAX_FOOD_SOURCES", type is "food"
            # Need to map them, e.g. by lowercasing and removing "max_" and "_sources"
            type_key = resource_type.lower().replace("max_", "").replace("_sources", "")
            if type_key == "murky_water": # Special case for naming consistency if needed
                type_key = "murky_water"
            elif type_key.endswith("s"): # e.g. woods -> wood
                 type_key = type_key[:-1]


            assert spawned_resource_counts[type_key] == expected_count, \
                f"Incorrect count for resource type '{type_key}'. Expected: {expected_count}, Got: {spawned_resource_counts[type_key]}"
            expected_total_resources += expected_count
            
        assert len(self.env.resources) == expected_total_resources, \
            f"Total number of spawned resources ({len(self.env.resources)}) does not match expected total ({expected_total_resources})."

    def test_no_overlapping_entities_on_reset(self):
        """Verify no entities (agent, resources) overlap on the grid."""
        self.env.reset()
        
        entity_coordinates = []
        
        # Agent coordinates
        agent_coord = (self.env.agent.x, self.env.agent.y)
        entity_coordinates.append(agent_coord)
        assert get_cell_content(self.env.grid, self.env.agent.x, self.env.agent.y) == AGENT, \
            "Agent's cell on grid does not contain AGENT marker."

        # Resource coordinates
        for resource in self.env.resources:
            res_coord = (resource.x, resource.y)
            entity_coordinates.append(res_coord)
            assert get_cell_content(self.env.grid, resource.x, resource.y) == RESOURCE, \
                f"Resource {resource.type} at ({resource.x},{resource.y}) not marked as RESOURCE on grid."

        # Check for duplicates
        coordinate_counts = Counter(entity_coordinates)
        for coord, count in coordinate_counts.items():
            assert count == 1, f"Coordinate {coord} is occupied by {count} entities (should be 1)."
            
    def test_resources_are_distinct_objects(self):
        """Verify all Resource objects are distinct instances."""
        self.env.reset()
        if not self.env.resources:
            pytest.skip("No resources spawned to test for distinctness.")
            
        resource_ids = [id(r) for r in self.env.resources]
        assert len(resource_ids) == len(set(resource_ids)), "Not all resource objects are distinct instances (same ID found)."

    def test_grid_consistent_with_resources_list(self):
        """Verify grid accurately reflects the self.resources list."""
        self.env.reset()
        
        grid_resource_count = 0
        for r_idx, row in enumerate(self.env.grid):
            for c_idx, cell_content in enumerate(row):
                if cell_content == RESOURCE:
                    grid_resource_count += 1
                    # Check if this grid resource is in self.resources list
                    assert any(res.x == c_idx and res.y == r_idx for res in self.env.resources), \
                        f"Grid shows RESOURCE at ({c_idx},{r_idx}), but no such resource in self.resources list."
                        
        assert grid_resource_count == len(self.env.resources), \
            f"Number of RESOURCE cells on grid ({grid_resource_count}) does not match length of self.resources ({len(self.env.resources)})."

        for resource in self.env.resources:
            assert get_cell_content(self.env.grid, resource.x, resource.y) == RESOURCE, \
                f"Resource {resource.type} at ({resource.x},{resource.y}) in list is not marked as RESOURCE on grid."

    def test_reset_on_small_grid(self):
        """Test reset behavior on a grid potentially too small for all entities."""
        # Calculate total entities: 1 agent + sum of all resource counts
        total_entities_to_spawn = 1 + sum(RESOURCE_LIMITS.values())
        
        # Test on a grid that's exactly the size needed (if possible) or slightly smaller
        # This test is more about graceful handling than perfect placement if grid is too small
        # For a 3x3 grid (9 cells), if total_entities_to_spawn > 9, we expect issues.
        small_grid_size = 3 
        if total_entities_to_spawn > small_grid_size * small_grid_size:
             # If too many entities, the current reset logic with all_coords.pop() will try to place them
             # until all_coords is empty, and then print a warning.
             # We just want to ensure it doesn't crash.
             self.env = SierraEnv(grid_width=small_grid_size, grid_height=small_grid_size)
             # Capture stdout/stderr to check for warning if possible, or just check for no crash
             try:
                 self.env.reset()
                 # Check that agent is placed
                 assert self.env.agent is not None
                 # Check that number of resources is <= grid_size*grid_size - 1
                 assert len(self.env.resources) <= (small_grid_size * small_grid_size) -1

             except ValueError as e:
                 # If grid is so small not even agent can be placed (e.g. 0x0), ValueError is expected
                 assert "Grid is too small to place agent" in str(e) or "Not enough space to place all resources" in str(e) # Should be agent error first
             except Exception as e:
                 pytest.fail(f"Reset on small grid {small_grid_size}x{small_grid_size} failed unexpectedly: {e}")

        else:
            # If all entities can fit, just ensure it runs without error
            self.env = SierraEnv(grid_width=small_grid_size, grid_height=small_grid_size)
            self.env.reset()
            assert self.env.agent is not None
            assert len(self.env.resources) == total_entities_to_spawn -1


    def test_reset_multiple_times(self):
        """Ensure reset can be called multiple times, clearing and re-populating correctly."""
        obs1, _ = self.env.reset()
        agent1_pos = (self.env.agent.x, self.env.agent.y)
        resources1_coords = sorted([(r.x, r.y, r.type) for r in self.env.resources])
        num_resources1 = len(self.env.resources)

        obs2, _ = self.env.reset()
        agent2_pos = (self.env.agent.x, self.env.agent.y)
        resources2_coords = sorted([(r.x, r.y, r.type) for r in self.env.resources])
        num_resources2 = len(self.env.resources)

        assert num_resources1 == num_resources2, "Number of resources should be consistent across resets."
        
        # Check that agent position or resource configuration is likely different (randomness)
        # This is not a strict guarantee but a probabilistic check for non-determinism.
        # A better check would be to ensure all entities are re-placed and grid is consistent.
        if num_resources1 > 0 and self.env.grid_width * self.env.grid_height > num_resources1 +1 : # Only if there's space for variation
             assert agent1_pos != agent2_pos or resources1_coords != resources2_coords, \
                 "Agent and resource configurations should likely change after reset (unless grid is tiny)."
        
        # Verify consistency after second reset
        self.test_agent_placed_on_reset() # Re-uses the first test's logic for the current state
        self.test_correct_number_of_resources_spawned()
        self.test_no_overlapping_entities_on_reset()
        self.test_grid_consistent_with_resources_list()
