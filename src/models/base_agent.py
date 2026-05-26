from src.configs.nn_config import NNOutputs

class BaseAgent:
    """Base mathematical model for all creatures in the simulation."""
    
    def __init__(self, x: float, y: float, speed: float, sprint_speed: float, radius: float, max_energy: float):
        self.x = x
        self.y = y
        self.base_speed = speed
        self.sprint_speed = sprint_speed
        self.radius = radius
        self.is_alive = True
        
        self.max_energy = max_energy
        self.current_energy = max_energy
        
        self.max_hunger = 100.0
        self.current_hunger = self.max_hunger
        self.hunger_consumption = 0.5

        self.energy_consumption_rate = 10
        self.energy_regen_rate = 4

        self.kills = 0
        self.kills_before_step = 0

        self.foods = 0
        self.foods_before_step = 0
        
        # Fitness tracking
        self.survival_frames = 0
        self.wall_frames = 0

    def apply_action(self, action_vector: NNOutputs, max_width: int, max_height: int):
        move_x = action_vector.x_direction
        move_y = action_vector.y_direction
        wants_to_sprint = action_vector.do_sprint > 0.5
        
        magnitude = (move_x**2 + move_y**2) ** 0.5
        
        if magnitude > 0:
            move_x /= magnitude
            move_y /= magnitude
        
        if wants_to_sprint and self.current_energy > 0:
            current_speed = self.sprint_speed
            self.current_energy = max(0, self.current_energy - self.energy_consumption_rate)
        else:
            current_speed = self.base_speed
            self.current_energy = min(self.max_energy, self.current_energy + self.energy_regen_rate)
        
        self.x += move_x * current_speed
        self.y += move_y * current_speed
        
        self.x = max(self.radius, min(self.x, max_width - self.radius))
        self.y = max(self.radius, min(self.y, max_height - self.radius))

        is_on_wall = (
            self.x <= self.radius + 1 or self.x >= max_width - self.radius - 1 or
            self.y <= self.radius + 1 or self.y >= max_height - self.radius - 1
        )
        if is_on_wall:
            self.wall_frames += 1