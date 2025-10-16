"""Manager classes that coordinate resource, threat, and time behaviour."""
from __future__ import annotations

from typing import List, Set, Tuple

from .constants import (
    ENVIRONMENT_CYCLE_CONSTANTS,
    GAMEPLAY_CONSTANTS,
    RESOURCE_LIMITS,
    TIME_CONSTANTS,
)
from .entities import Resource, Threat
from .grid import EMPTY, RESOURCE, THREAT, check_boundaries, get_cell_content, place_entity


def _normalise_resource_key(key: str) -> str:
    key = key.lower().removeprefix("max_")
    if key.endswith("_sources"):
        key = key[: -len("_sources")]
    if key.endswith("s") and not key.endswith("ss"):
        key = key[:-1]
    return key


class ResourceManager:
    def __init__(self, env: "SierraEnv") -> None:  # type: ignore[name-defined]
        self.env = env

    def reset(self) -> None:
        self.env.resources = []
        spawn_list: List[str] = []
        for key, count in RESOURCE_LIMITS.items():
            if key == "MAX_THREATS" or count <= 0:
                continue
            resource_type = _normalise_resource_key(key)
            spawn_list.extend([resource_type] * count)

        for resource_type in spawn_list:
            if not self.env._available_cells:
                break
            x, y = self.env._available_cells.pop()
            resource = Resource(x, y, resource_type)
            self.env.resources.append(resource)
            place_entity(self.env.grid, resource, RESOURCE, x=x, y=y)

    def remove_resource(self, resource: Resource) -> None:
        if resource in self.env.resources:
            self.env.resources.remove(resource)
        if check_boundaries(self.env.grid, resource.x, resource.y):
            self.env.grid[resource.y, resource.x] = EMPTY

    def step(self) -> None:
        # Respawn behaviour is intentionally left simple for now.
        return


class ThreatManager:
    def __init__(self, env: "SierraEnv") -> None:  # type: ignore[name-defined]
        self.env = env

    def reset(self) -> None:
        self.env.threats = []
        threat_count = RESOURCE_LIMITS.get("MAX_THREATS", 0)
        for _ in range(threat_count):
            if not self.env._available_cells:
                break
            x, y = self.env._available_cells.pop()
            threat = Threat(x, y)
            self.env.threats.append(threat)
            place_entity(self.env.grid, threat, THREAT, x=x, y=y)

    def _find_alternative_position(self, forbidden: Set[Tuple[int, int]]) -> Tuple[int, int] | None:
        for y in range(self.env.grid_height):
            for x in range(self.env.grid_width):
                if (x, y) in forbidden:
                    continue
                if get_cell_content(self.env.grid, x, y) == EMPTY:
                    return x, y
        return None

    def step(self, forbidden: Set[Tuple[int, int]] | None = None) -> bool:
        collision = False
        forbidden = forbidden or set()
        for threat in self.env.threats:
            previous_x, previous_y = threat.x, threat.y
            self.env.grid[previous_y, previous_x] = EMPTY

            threat.move_towards(self.env.agent.x, self.env.agent.y)

            if not check_boundaries(self.env.grid, threat.x, threat.y):
                threat.x = min(max(threat.x, 0), self.env.grid_width - 1)
                threat.y = min(max(threat.y, 0), self.env.grid_height - 1)

            if (threat.x, threat.y) in forbidden or get_cell_content(self.env.grid, threat.x, threat.y) == RESOURCE:
                alternative = self._find_alternative_position(forbidden)
                if alternative is not None:
                    threat.x, threat.y = alternative
                else:
                    threat.x, threat.y = previous_x, previous_y

            if threat.x == self.env.agent.x and threat.y == self.env.agent.y:
                collision = True
                place_entity(self.env.grid, threat, THREAT, x=threat.x, y=threat.y)
                continue

            place_entity(self.env.grid, threat, THREAT, x=threat.x, y=threat.y)
        return collision


class TimeManager:
    def __init__(self, env: "SierraEnv") -> None:  # type: ignore[name-defined]
        self.env = env

    def reset(self) -> None:
        self.env.world_time = 0
        self.env.is_day = True
        self._update_weather(force_change=True)
        self._update_season(force_change=True)

    def _update_weather(self, force_change: bool = False) -> None:
        weather_types = ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TYPES"]
        current = getattr(self.env, "current_weather", None)
        candidates = [w for w in weather_types if w != current]
        if force_change and candidates:
            self.env.current_weather = self.env.np_random.choice(candidates)
        else:
            self.env.current_weather = self.env.np_random.choice(candidates or weather_types)

    def _update_season(self, force_change: bool = False) -> None:
        season_types = ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TYPES"]
        current = getattr(self.env, "current_season", None)
        candidates = [s for s in season_types if s != current]
        if force_change and candidates:
            self.env.current_season = self.env.np_random.choice(candidates)
        else:
            self.env.current_season = self.env.np_random.choice(candidates or season_types)

    def step(self) -> None:
        cycle_length = TIME_CONSTANTS["DAY_LENGTH"] + TIME_CONSTANTS["NIGHT_LENGTH"]
        cycle_position = self.env.world_time % cycle_length
        self.env.is_day = cycle_position < TIME_CONSTANTS["DAY_LENGTH"]

        self.env.world_time += 1

        if (
            self.env.world_time % ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TRANSITION_STEPS"] == 0
        ):
            self._update_weather()

        if (
            self.env.world_time % ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TRANSITION_STEPS"] == 0
        ):
            self._update_season()
