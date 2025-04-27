class Agent:
    """Represents an agent in the environment."""
    def __init__(self, x, y, energy=100):
        self.x = x
        self.y = y
        self.energy = energy

class Resource:
    """Represents a resource in the environment."""
    def __init__(self, x, y, type='food'):
        self.x = x
        self.y = y
        self.type = type