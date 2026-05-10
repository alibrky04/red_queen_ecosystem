from dataclasses import dataclass

@dataclass
class NNInputs:
    enemy_x: float
    enemy_y: float
    enemy_distance: float
    current_energy: float
    dist_to_left: float
    dist_to_right: float
    dist_to_top: float
    dist_to_bottom: float

@dataclass
class NNOutputs:
    x_direction: float
    y_direction: float
    do_sprint: float