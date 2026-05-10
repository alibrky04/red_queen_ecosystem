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

        self.energy_consumption_rate = 10
        self.energy_regen_rate = 4
        
        # Fitness tracking
        self.survival_frames = 0
        self.total_distance = 0.0
        self.closest_enemy_distance = float('inf')

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
        
        # Track distance traveled
        self.total_distance += magnitude * current_speed
        
        self.x += move_x * current_speed
        self.y += move_y * current_speed
        
        # Clamp to borders
        self.x = max(self.radius, min(self.x, max_width - self.radius))
        self.y = max(self.radius, min(self.y, max_height - self.radius))