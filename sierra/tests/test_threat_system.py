
import pytest
from sierra.environment.core import SierraEnv, RESOURCE_LIMITS
from sierra.environment.grid import THREAT

class TestThreatSystem:

    def setup_method(self):
        self.env = SierraEnv()
        self.env.reset()

    def test_threat_spawn(self):
        assert len(self.env.threats) == RESOURCE_LIMITS["MAX_THREATS"]
        for threat in self.env.threats:
            assert self.env.grid[threat.y, threat.x] == THREAT

    def test_threat_movement(self):
        initial_threat_positions = [(threat.x, threat.y) for threat in self.env.threats]
        self.env.step(0) # Action doesn't matter
        new_threat_positions = [(threat.x, threat.y) for threat in self.env.threats]
        assert initial_threat_positions != new_threat_positions

    def test_agent_threat_collision(self):
        self.env.reset()
        # Place a threat next to the agent
        agent_x, agent_y = self.env.agent.x, self.env.agent.y
        threat_x, threat_y = agent_x + 1, agent_y
        
        # Remove existing threats and place a new one
        for threat in self.env.threats:
            self.env.grid[threat.y, threat.x] = 0
        self.env.threats = []
        
        if threat_x < self.env.grid_width:
            threat = self.env.threats.append(self.env.threats[0])
            threat.x = threat_x
            threat.y = threat_y
            self.env.grid[threat_y, threat_x] = THREAT

            initial_hunger = self.env.agent.hunger
            initial_thirst = self.env.agent.thirst

            # Move agent into the threat
            self.env.step(3) # Move right

            assert self.env.agent.hunger < initial_hunger
            assert self.env.agent.thirst < initial_thirst
