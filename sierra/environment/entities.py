class Agent:
    """Represents an agent in the environment."""
    def __init__(self, x, y, energy=100, hunger=100, thirst=100):
        self.x = x
        self.y = y
        self.energy = energy
        self.hunger = hunger
        self.thirst = thirst
        self.inventory = {material: 0 for material in Resource.MATERIAL_TYPES}
        self.has_shelter = False
        self.has_axe = False
        self.water_filters_available = 0

class Resource:
    """Represents a resource in the environment."""
    MATERIAL_TYPES = ['wood', 'stone', 'charcoal', 'cloth', 'murky_water']

    def __init__(self, x, y, type='food'):
        self.x = x
        self.y = y
        if type not in ['food', 'water'] + self.MATERIAL_TYPES:
            raise ValueError(f"Invalid resource type: {type}")
        self.type = type