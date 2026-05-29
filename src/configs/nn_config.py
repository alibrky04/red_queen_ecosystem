from dataclasses import dataclass

@dataclass
class NNInputs:
    enemy_sector_distances: list  # 8 values: distance to closest enemy in each sector (0.0 to 1.0)
    enemy_sector_velocities: list # 8 values: velocity magnitude of enemy in each sector (0.0 to 1.0)
    enemy_count: float            # Total enemy count in FOV (normalized)
    closest_enemy_dx: float       # Normalized direction to closest enemy (-1.0 to 1.0)
    closest_enemy_dy: float       # Normalized direction to closest enemy (-1.0 to 1.0)
    closest_enemy_distance: float # Distance to closest enemy (0.0 to 1.0)

    current_energy: float         # 0.0 to 1.0
    current_hunger: float         # 0.0 to 1.0
    
    food_dx: float                # -1.0 to 1.0
    food_dy: float                # -1.0 to 1.0
    food_distance: float          # 0.0 to 1.0
    
    prox_left: float              # 0.0 (safe) to 1.0 (touching)
    prox_right: float             # 0.0 (safe) to 1.0 (touching)
    prox_top: float               # 0.0 (safe) to 1.0 (touching)
    prox_bottom: float            # 0.0 (safe) to 1.0 (touching)

@dataclass
class NNOutputs:
    x_direction: float
    y_direction: float
    do_sprint: float