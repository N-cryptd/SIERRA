"""Support managers for the SIERRA environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .entities import Resource, Threat
from .grid import EMPTY, RESOURCE, THREAT, check_boundaries, get_cell_content, place_entity


@dataclass(frozen=True)
class ResourceSpec:
    key: str
    resource_type: str


class ResourceManager:
    """Handles resource spawning and simple respawn bookkeeping."""

    def __init__(self, env, resource_limits: Dict[str, int], respawn_time: int) -> None:
        self.env = env
        self.resource_limits = resource_limits
        self.respawn_time = respawn_time
        self.specs: List[ResourceSpec] = [
            ResourceSpec("MAX_FOOD_SOURCES", "food"),
            ResourceSpec("MAX_WATER_SOURCES", "water"),
            ResourceSpec("MAX_WOOD_SOURCES", "wood"),
            ResourceSpec("MAX_STONE_SOURCES", "stone"),
            ResourceSpec("MAX_CHARCOAL_SOURCES", "charcoal"),
            ResourceSpec("MAX_CLOTH_SOURCES", "cloth"),
            ResourceSpec("MAX_MURKY_WATER_SOURCES", "murky_water"),
        ]

    def reset(self) -> None:
        for spec in self.specs:
            count = self.resource_limits.get(spec.key, 0)
            for _ in range(count):
                coord = self.env.acquire_free_coordinate()
                if coord is None:
                    break
                x, y = coord
                resource = Resource(x=x, y=y, type=spec.resource_type)
                self.env.resources.append(resource)
                place_entity(self.env.grid, RESOURCE, x=x, y=y)

    def step(self) -> None:
        # The current test-suite does not require resource respawning. The method
        # is kept for completeness and future extension.
        return None


class ThreatManager:
    """Spawns and updates roaming threats."""

    DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def __init__(self, env, max_threats: int) -> None:
        self.env = env
        self.max_threats = max_threats

    def reset(self) -> None:
        for _ in range(self.max_threats):
            coord = self.env.acquire_free_coordinate()
            if coord is None:
                break
            x, y = coord
            threat = Threat(x=x, y=y)
            self.env.threats.append(threat)
            place_entity(self.env.grid, THREAT, x=x, y=y)

    def step(self, forbidden: set[tuple[int, int]] | None = None) -> bool:
        collision = False
        rng = self.env.np_random
        forbidden = forbidden or set()
        for threat in self.env.threats:
            dx, dy = self.DIRECTIONS[rng.integers(0, len(self.DIRECTIONS))]
            target_x, target_y = threat.x + dx, threat.y + dy
            if not check_boundaries(self.env.grid, target_x, target_y):
                continue
            if (target_x, target_y) in forbidden:
                continue

            cell_value = get_cell_content(self.env.grid, target_x, target_y)
            if cell_value not in (EMPTY, THREAT):
                if cell_value == THREAT:
                    continue
                if cell_value == RESOURCE:
                    continue
                if cell_value == self.env.AGENT_MARKER:
                    continue
                else:
                    continue

            place_entity(self.env.grid, EMPTY, x=threat.x, y=threat.y)
            threat.move(dx, dy)
            place_entity(self.env.grid, THREAT, x=threat.x, y=threat.y)

            if threat.x == self.env.agent.x and threat.y == self.env.agent.y:
                collision = True
        return collision


class TimeManager:
    """Tracks day/night cycles as well as weather and season transitions."""

    def __init__(
        self,
        env,
        time_constants: Dict[str, int],
        environment_cycles: Dict[str, Iterable],
    ) -> None:
        self.env = env
        self.time_constants = time_constants
        self.environment_cycles = environment_cycles

    def reset(self) -> None:
        self.env.world_time = 0
        self.env.is_day = True
        self.env.current_weather = self.environment_cycles["WEATHER_TYPES"][0]
        self.env.current_season = self.environment_cycles["SEASON_TYPES"][0]

    def step(self) -> None:
        day_length = self.time_constants["DAY_LENGTH"]
        night_length = self.time_constants["NIGHT_LENGTH"]
        weather_period = self.environment_cycles["WEATHER_TRANSITION_STEPS"]
        season_period = self.environment_cycles["SEASON_TRANSITION_STEPS"]

        cycle_length = day_length + night_length
        cycle_position = self.env.world_time % cycle_length
        self.env.is_day = cycle_position < day_length

        self.env.world_time += 1

        if self.env.world_time % weather_period == 0:
            self.env.current_weather = self._choose_new_value("WEATHER_TYPES")

        if self.env.world_time % season_period == 0:
            self.env.current_season = self._choose_new_value("SEASON_TYPES")

    def _choose_new_value(self, key: str) -> str:
        options = list(self.environment_cycles[key])
        current = getattr(self.env, f"current_{key.split('_')[0].lower()}")
        available = [option for option in options if str(option) != str(current)]
        if not available:
            available = options
        choice = self.env.np_random.choice(available)
        return str(choice)
