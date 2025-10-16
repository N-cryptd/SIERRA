from __future__ import annotations

import random

from .entities import Resource, Threat
from .grid import EMPTY, RESOURCE, THREAT, check_boundaries, get_cell_content, place_entity


def _resource_type_from_limit_key(limit_key: str) -> str:
    """Normalise configuration keys to resource type names."""

    normalized = limit_key.removeprefix("MAX_").removesuffix("_SOURCES").lower()
    if normalized == "murky_water":
        return normalized
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized

class ResourceManager:
    def __init__(self, env):
        self.env = env
        self.depleted_resources: list[Resource] = []

    def reset(self):
        # Define resource counts based on config and season
        resource_counts_dict = self.env.config['resource_limits'].copy()

        self.depleted_resources = []

        resource_types_to_spawn = []
        for res_type, count in resource_counts_dict.items():
            if res_type == "MAX_THREATS":
                continue
            resource_types_to_spawn.extend([_resource_type_from_limit_key(res_type)] * count)
        
        for res_type in resource_types_to_spawn:
            if not self.env.all_coords:
                print(f"Warning: Not enough space to place all resources. Grid size: {self.env.grid_width}x{self.env.grid_height}, trying to place {res_type}")
                break 
            
            res_x, res_y = self.env.all_coords.pop()
            new_resource = Resource(res_x, res_y, type=res_type)
            self.env.resources.append(new_resource)
            place_entity(self.env.grid, new_resource, RESOURCE, x=new_resource.x, y=new_resource.y)

    def step(self):
        self._handle_resource_respawn()

    def mark_resource_depleted(self, resource: Resource) -> None:
        """Register a resource for respawn and remove it from active lists."""

        if resource in self.env.resources:
            self.env.resources.remove(resource)
        resource.respawn_timer = self.env.config['gameplay']['RESOURCE_RESPAWN_TIME']
        if resource not in self.depleted_resources:
            self.depleted_resources.append(resource)

    def _handle_resource_respawn(self):
        """Handles the respawning of resources."""
        for resource in list(self.depleted_resources):
            if resource.respawn_timer > 0:
                resource.respawn_timer -= 1
            if resource.respawn_timer == 0 and self.env.grid[resource.y, resource.x] == EMPTY:
                place_entity(self.env.grid, resource, RESOURCE, x=resource.x, y=resource.y)
                self.env.resources.append(resource)
                self.depleted_resources.remove(resource)

class ThreatManager:
    def __init__(self, env):
        self.env = env

    def reset(self):
        max_threats = self.env.config.get("max_threats", self.env.config['resource_limits']["MAX_THREATS"])
        for _ in range(max_threats):
            if not self.env.all_coords:
                print(f"Warning: Not enough space to place all threats. Grid size: {self.env.grid_width}x{self.env.grid_height}")
                break
            
            threat_x, threat_y = self.env.all_coords.pop()
            new_threat = Threat(threat_x, threat_y)
            self.env.threats.append(new_threat)
            place_entity(self.env.grid, new_threat, THREAT, x=new_threat.x, y=new_threat.y)

    def step(self):
        return self._handle_threats()

    def _handle_threats(self):
        """Moves threats and checks for collisions with the agent."""
        collision = False
        for threat in self.env.threats:
            if self.env.has_line_of_sight((threat.x, threat.y), (self.env.agent.x, self.env.agent.y)):
                threat.state = "CHASING"
                threat.target = (self.env.agent.x, self.env.agent.y)
            else:
                threat.state = "PATROLLING"
                threat.target = None

            if threat.state == "CHASING":
                move_x, move_y = 0, 0
                if threat.target[0] > threat.x:
                    move_x = 1
                elif threat.target[0] < threat.x:
                    move_x = -1
                if threat.target[1] > threat.y:
                    move_y = 1
                elif threat.target[1] < threat.y:
                    move_y = -1
            else:
                move_x, move_y = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            
            new_x, new_y = threat.x + move_x, threat.y + move_y

            if check_boundaries(self.env.grid, new_x, new_y) and get_cell_content(self.env.grid, new_x, new_y) == EMPTY:
                self.env.grid[threat.y, threat.x] = EMPTY
                threat.x, threat.y = new_x, new_y
                place_entity(self.env.grid, threat, THREAT, x=threat.x, y=threat.y)

        # Check for collision with agent
        for threat in self.env.threats:
            if threat.x == self.env.agent.x and threat.y == self.env.agent.y:
                collision = True
        
        return collision

class TimeManager:
    def __init__(self, env):
        self.env = env

    def reset(self):
        self.env.world_time = 0
        self.env.is_day = True
        self.env.current_weather = self.env.np_random.choice(self.env.config['environment_cycles']["WEATHER_TYPES"])
        self.env.current_season = self.env.np_random.choice(self.env.config['environment_cycles']["SEASON_TYPES"])

    def step(self):
        self.env.world_time += 1
        cycle_length = (
            self.env.config['time_constants']["DAY_LENGTH"]
            + self.env.config['time_constants']["NIGHT_LENGTH"]
        )
        cycle_position = (self.env.world_time - 1) % cycle_length
        self.env.is_day = cycle_position < self.env.config['time_constants']["DAY_LENGTH"]

        if (
            self.env.world_time
            % self.env.config['environment_cycles']["WEATHER_TRANSITION_STEPS"]
            == 0
        ):
            weather_options = [
                w
                for w in self.env.config['environment_cycles']["WEATHER_TYPES"]
                if w != self.env.current_weather
            ]
            if not weather_options:
                weather_options = self.env.config['environment_cycles']["WEATHER_TYPES"]
            self.env.current_weather = self.env.np_random.choice(weather_options)

        if (
            self.env.world_time
            % self.env.config['environment_cycles']["SEASON_TRANSITION_STEPS"]
            == 0
        ):
            season_options = [
                s
                for s in self.env.config['environment_cycles']["SEASON_TYPES"]
                if s != self.env.current_season
            ]
            if not season_options:
                season_options = self.env.config['environment_cycles']["SEASON_TYPES"]
            self.env.current_season = self.env.np_random.choice(season_options)