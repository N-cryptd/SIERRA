"""Helper manager classes that keep the environment logic tidy."""

from __future__ import annotations

from .constants import (
    ENVIRONMENT_CYCLE_CONSTANTS,
    GAMEPLAY_CONSTANTS,
    RESOURCE_LIMITS,
    TIME_CONSTANTS,
    WEATHER_TYPES,
    SEASON_TYPES,
    limit_key_to_resource,
)
from .entities import Resource, Threat
from .grid import EMPTY, RESOURCE, THREAT, check_boundaries, get_cell_content, place_entity


class ResourceManager:
    def __init__(self, env: "SierraEnv"):
        self.env = env

    def reset(self) -> None:
        self.env.resources = []
        for limit_name, count in RESOURCE_LIMITS.items():
            resource_type = limit_key_to_resource(limit_name)
            for _ in range(count):
                if not self.env._available_coords:
                    return
                x, y = self.env._available_coords.pop()
                resource = Resource(x, y, resource_type)
                self.env.resources.append(resource)
                place_entity(self.env.grid, resource, RESOURCE, x=x, y=y)

    def step(self) -> None:
        respawn_time = GAMEPLAY_CONSTANTS.get("RESOURCE_RESPAWN_TIME", 0)
        if respawn_time <= 0:
            return
        for resource in self.env.resources:
            if not isinstance(resource, Resource):
                continue
            if resource.respawn_timer > 0:
                resource.respawn_timer -= 1
                if resource.respawn_timer == 0 and get_cell_content(self.env.grid, resource.x, resource.y) == EMPTY:
                    place_entity(self.env.grid, resource, RESOURCE)


class ThreatManager:
    def __init__(self, env: "SierraEnv"):
        self.env = env

    def reset(self) -> None:
        self.env.threats = []
        count = RESOURCE_LIMITS.get("MAX_THREATS", 0)
        for _ in range(count):
            if not self.env._available_coords:
                return
            index = None
            for i, (candidate_x, candidate_y) in enumerate(self.env._available_coords):
                if abs(candidate_x - self.env.agent.x) + abs(candidate_y - self.env.agent.y) > 1:
                    index = i
                    break
            if index is None:
                x, y = self.env._available_coords.pop()
            else:
                x, y = self.env._available_coords.pop(index)
            threat = Threat(x, y)
            self.env.threats.append(threat)
            place_entity(self.env.grid, threat, THREAT, x=x, y=y)

    def step(self) -> bool:
        collision = False
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for threat in self.env.threats:
            moved = False
            shuffled = directions[:]
            self.env.np_random.shuffle(shuffled)
            for move_x, move_y in shuffled:
                new_x, new_y = threat.x + move_x, threat.y + move_y
                if not check_boundaries(self.env.grid, new_x, new_y):
                    continue
                if (new_x, new_y) == (self.env.agent.x, self.env.agent.y):
                    continue
                if (new_x, new_y) == getattr(self.env, "_last_agent_position", None):
                    continue
                if get_cell_content(self.env.grid, new_x, new_y) != EMPTY:
                    continue
                self.env.grid[threat.y, threat.x] = EMPTY
                threat.x, threat.y = new_x, new_y
                place_entity(self.env.grid, threat, THREAT)
                moved = True
                break
            if not moved and (threat.x, threat.y) == (self.env.agent.x, self.env.agent.y):
                collision = True
        return collision


class TimeManager:
    def __init__(self, env: "SierraEnv"):
        self.env = env

    def reset(self) -> None:
        self.env.world_time = 0
        self.env.is_day = True
        self.env.current_weather = self.env.np_random.choice(WEATHER_TYPES) if WEATHER_TYPES else "clear"
        self.env.current_season = self.env.np_random.choice(SEASON_TYPES) if SEASON_TYPES else "spring"

    def step(self) -> None:
        day_length = TIME_CONSTANTS.get("DAY_LENGTH", 1)
        night_length = TIME_CONSTANTS.get("NIGHT_LENGTH", 1)
        cycle = day_length + night_length
        if cycle > 0:
            self.env.is_day = (self.env.world_time % cycle) < day_length

        new_time = self.env.world_time + 1

        weather_steps = ENVIRONMENT_CYCLE_CONSTANTS.get("WEATHER_TRANSITION_STEPS", 0)
        if weather_steps and WEATHER_TYPES and new_time % weather_steps == 0:
            choices = [w for w in WEATHER_TYPES if w != self.env.current_weather]
            self.env.current_weather = self.env.np_random.choice(choices or WEATHER_TYPES)

        season_steps = ENVIRONMENT_CYCLE_CONSTANTS.get("SEASON_TRANSITION_STEPS", 0)
        if season_steps and SEASON_TYPES and new_time % season_steps == 0:
            choices = [s for s in SEASON_TYPES if s != self.env.current_season]
            self.env.current_season = self.env.np_random.choice(choices or SEASON_TYPES)

        self.env.world_time = new_time


__all__ = ["ResourceManager", "ThreatManager", "TimeManager"]
