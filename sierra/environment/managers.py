from .grid import place_entity, get_cell_content, EMPTY, RESOURCE, THREAT
from .entities import Resource, Threat
import random

class ResourceManager:
    def __init__(self, env):
        self.env = env

    def reset(self):
        # Define resource counts based on config and season
        resource_counts_dict = self.env.config['resource_limits'].copy()

        if self.env.current_season == 'winter':
            resource_counts_dict['MAX_FOOD_SOURCES'] = 1
            resource_counts_dict['MAX_WATER_SOURCES'] = 0
        
        resource_types_to_spawn = []
        for res_type, count in resource_counts_dict.items():
            resource_types_to_spawn.extend([res_type.replace("MAX_", "").replace("_SOURCES", "").lower()] * count)
        
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

    def _handle_resource_respawn(self):
        """Handles the respawning of resources."""
        for resource in self.env.resources:
            if resource.respawn_timer > 0:
                resource.respawn_timer -= 1
                if resource.respawn_timer == 0:
                    if self.env.grid[resource.y, resource.x] == EMPTY:
                        place_entity(self.env.grid, resource, RESOURCE, x=resource.x, y=resource.y)

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

            if self.env.check_boundaries(new_x, new_y) and get_cell_content(self.env.grid, new_x, new_y) == EMPTY:
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
        cycle_length = self.env.config['time_constants']["DAY_LENGTH"] + self.env.config['time_constants']["NIGHT_LENGTH"]
        self.env.is_day = (self.env.world_time % cycle_length) < self.env.config['time_constants']["DAY_LENGTH"]

        if self.env.world_time % self.env.config['environment_cycles']["WEATHER_TRANSITION_STEPS"] == 0:
             self.env.current_weather = self.env.np_random.choice(self.env.config['environment_cycles']["WEATHER_TYPES"])

        if self.env.world_time % self.env.config['environment_cycles']["SEASON_TRANSITION_STEPS"] == 0:
             self.env.current_season = self.env.np_random.choice(self.env.config['environment_cycles']["SEASON_TYPES"])