
import pytest
from sierra.environment.core import SierraEnv, TIME_CONSTANTS, ENVIRONMENT_CYCLE_CONSTANTS, GAMEPLAY_CONSTANTS

class TestDynamicEnvironment:

    def setup_method(self):
        self.env = SierraEnv()
        self.env.reset()

    def test_day_night_cycle(self):
        # Initial state should be day
        assert self.env.is_day is True

        # Advance time to just before night
        self.env.world_time = TIME_CONSTANTS["DAY_LENGTH"] - 1
        self.env.step(0) # Action doesn't matter
        assert self.env.is_day is True

        # Advance time to the first step of night
        self.env.step(0)
        assert self.env.is_day is False

        # Advance time to just before day
        self.env.world_time = TIME_CONSTANTS["DAY_LENGTH"] + TIME_CONSTANTS["NIGHT_LENGTH"] - 1
        self.env.step(0)
        assert self.env.is_day is False

        # Advance time to the first step of the next day
        self.env.step(0)
        assert self.env.is_day is True

    def test_weather_transition(self):
        initial_weather = self.env.current_weather
        
        # Advance time to just before weather transition
        for _ in range(ENVIRONMENT_CYCLE_CONSTANTS["WEATHER_TRANSITION_STEPS"] - 1):
            self.env.step(0)
        
        assert self.env.current_weather == initial_weather

        # Advance one more step to trigger weather transition
        self.env.step(0)
        assert self.env.current_weather != initial_weather

    def test_season_transition(self):
        initial_season = self.env.current_season
        
        # Advance time to just before season transition
        for _ in range(ENVIRONMENT_CYCLE_CONSTANTS["SEASON_TRANSITION_STEPS"] - 1):
            self.env.step(0)
            
        assert self.env.current_season == initial_season

        # Advance one more step to trigger season transition
        self.env.step(0)
        assert self.env.current_season != initial_season

    def test_needs_decay_day_vs_night(self):
        # Day
        self.env.reset()
        self.env.is_day = True
        initial_hunger = self.env.agent.hunger
        initial_thirst = self.env.agent.thirst
        self.env.step(0)
        day_hunger_decay = initial_hunger - self.env.agent.hunger
        day_thirst_decay = initial_thirst - self.env.agent.thirst

        # Night
        self.env.reset()
        self.env.is_day = False
        initial_hunger = self.env.agent.hunger
        initial_thirst = self.env.agent.thirst
        self.env.step(0)
        night_hunger_decay = initial_hunger - self.env.agent.hunger
        night_thirst_decay = initial_thirst - self.env.agent.thirst

        assert night_hunger_decay > day_hunger_decay
        assert night_thirst_decay > day_thirst_decay

    def test_shelter_effect_on_needs_decay(self):
        # No Shelter
        self.env.reset()
        self.env.agent.has_shelter = False
        initial_hunger = self.env.agent.hunger
        initial_thirst = self.env.agent.thirst
        self.env.step(0)
        no_shelter_hunger_decay = initial_hunger - self.env.agent.hunger
        no_shelter_thirst_decay = initial_thirst - self.env.agent.thirst

        # With Shelter
        self.env.reset()
        self.env.agent.has_shelter = True
        initial_hunger = self.env.agent.hunger
        initial_thirst = self.env.agent.thirst
        self.env.step(0)
        shelter_hunger_decay = initial_hunger - self.env.agent.hunger
        shelter_thirst_decay = initial_thirst - self.env.agent.thirst

        assert shelter_hunger_decay < no_shelter_hunger_decay
        assert shelter_thirst_decay < no_shelter_thirst_decay
