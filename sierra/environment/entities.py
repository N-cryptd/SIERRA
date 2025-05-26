# Attempt to import constants. If this causes issues, they might need to be passed or defined locally.
# For now, let's assume these can be imported after core.py is also updated,
# or we use placeholders if direct import fails during testing.
try:
    from sierra.environment.core import INVENTORY_CONSTANTS, GAMEPLAY_CONSTANTS
    # Define defaults if specific keys are missing, as per instructions
    INITIAL_HUNGER = GAMEPLAY_CONSTANTS.get('INITIAL_HUNGER', 100.0)
    INITIAL_THIRST = GAMEPLAY_CONSTANTS.get('INITIAL_THIRST', 100.0)
    MAX_THIRST = GAMEPLAY_CONSTANTS.get('MAX_THIRST', 100.0) # Assuming max hunger is also 100 implicitly
    MAX_HUNGER = GAMEPLAY_CONSTANTS.get('MAX_HUNGER', 100.0)

    # Decay constants - these should ideally be in GAMEPLAY_CONSTANTS too
    BASE_DECAY = GAMEPLAY_CONSTANTS.get('BASE_DECAY', 0.1)
    SHELTER_MULTIPLIER = GAMEPLAY_CONSTANTS.get('SHELTER_MULTIPLIER', 0.75)
    NIGHT_MULTIPLIER = GAMEPLAY_CONSTANTS.get('NIGHT_MULTIPLIER', 1.2)
    NIGHT_NO_SHELTER_EXTRA_MULTIPLIER = GAMEPLAY_CONSTANTS.get('NIGHT_NO_SHELTER_EXTRA_MULTIPLIER', 1.5)

except ImportError: # Fallback if core.py isn't structured yet or causes circular import
    INVENTORY_CONSTANTS = {"MAX_INVENTORY_PER_ITEM": 10, "MAX_WATER_FILTERS": 5}
    INITIAL_HUNGER = 100.0
    INITIAL_THIRST = 100.0
    MAX_THIRST = 100.0
    MAX_HUNGER = 100.0
    BASE_DECAY = 0.1
    SHELTER_MULTIPLIER = 0.75
    NIGHT_MULTIPLIER = 1.2
    NIGHT_NO_SHELTER_EXTRA_MULTIPLIER = 1.5


class Agent:
    """Represents an agent in the environment."""
    def __init__(self, x, y, hunger=None, thirst=None): # hunger/thirst now use constants
        self.x = x
        self.y = y
        self.hunger = INITIAL_HUNGER if hunger is None else hunger
        self.thirst = INITIAL_THIRST if thirst is None else thirst
        self.inventory = {material: 0 for material in Resource.MATERIAL_TYPES}
        self.has_shelter = False
        self.has_axe = False
        self.axe_durability = 0
        self.water_filters_available = 0

    # --- Inventory Management ---
    def add_item(self, item_type: str, quantity: int = 1) -> bool:
        if item_type not in self.inventory:
            # This case should ideally not happen if inventory is initialized with all Resource.MATERIAL_TYPES
            # For robustness, we can add it, or return False if only known types are allowed.
            # Assuming only pre-defined material types are allowed in inventory.
            return False # Item type not recognized for inventory

        current_amount = self.inventory[item_type]
        if current_amount >= INVENTORY_CONSTANTS['MAX_INVENTORY_PER_ITEM']:
            return False # Already full for this item

        can_add = INVENTORY_CONSTANTS['MAX_INVENTORY_PER_ITEM'] - current_amount
        add_actual = min(quantity, can_add)
        
        if add_actual > 0:
            self.inventory[item_type] += add_actual
            return True
        return False # No space to add anything or quantity was zero

    def remove_item(self, item_type: str, quantity: int = 1) -> bool:
        if not self.has_item(item_type, quantity):
            return False
        self.inventory[item_type] -= quantity
        return True

    def has_item(self, item_type: str, quantity: int = 1) -> bool:
        return self.inventory.get(item_type, 0) >= quantity

    def get_item_count(self, item_type: str) -> int:
        return self.inventory.get(item_type, 0)

    # --- Needs Management ---
    def update_needs(self, is_day: bool, environment_has_shelter_effect: bool): # environment_has_shelter_effect is self.has_shelter
        hunger_decay_rate = BASE_DECAY
        thirst_decay_rate = BASE_DECAY

        if environment_has_shelter_effect: # Agent is benefiting from its shelter
            hunger_decay_rate *= SHELTER_MULTIPLIER
            thirst_decay_rate *= SHELTER_MULTIPLIER

        if not is_day: # Night effect
            hunger_decay_rate *= NIGHT_MULTIPLIER
            thirst_decay_rate *= NIGHT_MULTIPLIER
            if not environment_has_shelter_effect: # Additional penalty if no shelter at night
                 hunger_decay_rate *= NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
                 thirst_decay_rate *= NIGHT_NO_SHELTER_EXTRA_MULTIPLIER
        
        self.hunger = max(0.0, self.hunger - hunger_decay_rate)
        self.thirst = max(0.0, self.thirst - thirst_decay_rate)

    def replenish_thirst(self, amount: float):
        self.thirst = min(MAX_THIRST, self.thirst + amount)
        
    def replenish_hunger(self, amount: float): # Added for completeness, if food consumption is implemented
        self.hunger = min(MAX_HUNGER, self.hunger + amount)

    def is_dead(self) -> bool:
        return self.hunger <= 0 or self.thirst <= 0

    # --- Crafting Status Methods ---
    def set_has_shelter(self, value: bool):
        self.has_shelter = value

    def set_has_axe(self, value: bool):
        self.has_axe = value
        if value:
            self.axe_durability = 100

    def add_water_filter(self, count: int = 1) -> bool:
        if self.water_filters_available < INVENTORY_CONSTANTS['MAX_WATER_FILTERS']:
            can_add = INVENTORY_CONSTANTS['MAX_WATER_FILTERS'] - self.water_filters_available
            add_actual = min(count, can_add)
            if add_actual > 0:
                self.water_filters_available += add_actual
                return True
        return False

    def use_water_filter(self) -> bool:
        if self.water_filters_available > 0:
            self.water_filters_available -= 1
            return True
        return False

class Resource:
    """Represents a resource in the environment."""
    MATERIAL_TYPES = ['wood', 'stone', 'charcoal', 'cloth', 'murky_water', 'food', 'water', 'sharpening_stone'] # Added food and water for inventory tracking consistency

    def __init__(self, x, y, type='food'):
        self.x = x
        self.y = y
        if type not in ['food', 'water'] + self.MATERIAL_TYPES:
            raise ValueError(f"Invalid resource type: {type}")
        self.type = type
        self.respawn_timer = 0

class Threat:
    """Represents a threat in the environment."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.state = "PATROLLING"
        self.target = None