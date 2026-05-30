from src.configs.nn_config import NNOutputs

class BaseAgent:
    def __init__(self, x: float, y: float, speed: float, sprint_speed: float, radius: float, max_energy: float, max_hunger: float):
        self.x = x
        self.y = y
        self.base_speed = speed
        self.sprint_speed = sprint_speed
        self.radius = radius
        self.is_alive = True
        
        self.max_energy = max_energy
        self.current_energy = max_energy
        
        self.max_hunger = max_hunger
        self.current_hunger = max_hunger
        self.hunger_consumption = 0.5

        self.energy_consumption_rate = 10
        self.energy_regen_rate = 4

        self.kills = 0
        self.kills_before_step = 0

        self.foods = 0
        self.foods_before_step = 0

        self.vx = 0.0
        self.vy = 0.0

        self.momentum = 0.55
        
        # Fitness / behavior tracking
        self.survival_frames = 0
        self.wall_frames = 0
        self.idle_frames = 0
        self.jitter_frames = 0
        self.move_effort = 0.0
        self.last_move_dir = None

    def apply_action(self, action_vector: NNOutputs, max_width: int, max_height: int):
        def clamp(value: float, low: float, high: float) -> float:
            return max(low, min(value, high))

        target_x = clamp(action_vector.x_direction, -1.0, 1.0)
        target_y = clamp(action_vector.y_direction, -1.0, 1.0)
        wants_to_sprint = action_vector.do_sprint > 0.25

        deadzone = 0.08
        if abs(target_x) < deadzone:
            target_x = 0.0
        if abs(target_y) < deadzone:
            target_y = 0.0

        self.vx = self.vx * self.momentum + target_x * (1.0 - self.momentum)
        self.vy = self.vy * self.momentum + target_y * (1.0 - self.momentum)

        magnitude = (self.vx ** 2 + self.vy ** 2) ** 0.5

        if magnitude > 1.0:
            self.vx /= magnitude
            self.vy /= magnitude
            magnitude = 1.0

        if magnitude < deadzone:
            self.vx = 0.0
            self.vy = 0.0
            magnitude = 0.0

        if wants_to_sprint and self.current_energy > 0 and magnitude > 0.0:
            current_speed = self.sprint_speed
            self.current_energy = max(0, self.current_energy - self.energy_consumption_rate)
        else:
            current_speed = self.base_speed
            self.current_energy = min(self.max_energy, self.current_energy + self.energy_regen_rate)

        old_x = self.x
        old_y = self.y

        self.x += self.vx * current_speed
        self.y += self.vy * current_speed

        self.x = max(self.radius, min(self.x, max_width - self.radius))
        self.y = max(self.radius, min(self.y, max_height - self.radius))

        if self.x <= self.radius:
            self.vx = max(0.0, self.vx)
        if self.x >= max_width - self.radius:
            self.vx = min(0.0, self.vx)
        if self.y <= self.radius:
            self.vy = max(0.0, self.vy)
        if self.y >= max_height - self.radius:
            self.vy = min(0.0, self.vy)

        move_dx = self.x - old_x
        move_dy = self.y - old_y
        move_dist = (move_dx ** 2 + move_dy ** 2) ** 0.5

        self.move_effort += move_dist

        if move_dist < 0.05:
            self.idle_frames += 1
        else:
            dir_x = move_dx / move_dist
            dir_y = move_dy / move_dist

            if self.last_move_dir is not None:
                dot = dir_x * self.last_move_dir[0] + dir_y * self.last_move_dir[1]
                if dot < -0.25:
                    self.jitter_frames += 1

            self.last_move_dir = (dir_x, dir_y)

        if (
            self.x <= self.radius + 1
            or self.x >= max_width - self.radius - 1
            or self.y <= self.radius + 1
            or self.y >= max_height - self.radius - 1
        ):
            self.wall_frames += 1