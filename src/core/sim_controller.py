import math
from src.models.prey import Prey
from src.models.predator import Predator
from src.configs.nn_config import NNInputs

class SimulationController:
    """Manages the environment state, physics, and collisions."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.preys = []
        self.predators = []
        self.foods = []

    def update_physics(self):
        self._check_collisions()

    def _check_collisions(self):
        """Calculates distance between predators and prey to determine eating."""
        for predator in self.predators:
            for prey in self.preys:
                if not prey.is_alive:
                    continue
                
                dist = math.hypot(predator.x - prey.x, predator.y - prey.y)
                if dist < (predator.radius + prey.radius):
                    prey.is_alive = False
                    predator.kills += 1

        for prey in self.preys:
            for food in self.foods:
                dist = math.hypot(prey.x - food.x, prey.y - food.y)
                if dist < (prey.radius + food.radius):
                    self.foods.remove(food)
                    prey.foods += 1
                    
        self.preys = [p for p in self.preys if p.is_alive]
        self.predators = [p for p in self.predators if p.is_alive]
    
    def get_agent_inputs(self, agent, enemies, foods):
        """Calculates normalized sensory inputs for a given agent using sector-based FOV."""
        
        whisker_len = 200.0
        prox_left = max(0.0, 1.0 - (agent.x / whisker_len))
        prox_right = max(0.0, 1.0 - ((self.width - agent.x) / whisker_len))
        prox_top = max(0.0, 1.0 - (agent.y / whisker_len))
        prox_bottom = max(0.0, 1.0 - ((self.height - agent.y) / whisker_len))

        norm_energy = max(0, agent.current_energy) / agent.max_energy
        norm_hunger = max(0, agent.current_hunger) / agent.max_hunger

        fov_range = 300.0
        enemy_sector_distances = [1.0] * 8
        enemy_sector_velocities = [0.0] * 8
        enemy_count = 0.0
        closest_enemy_dx = 0.0
        closest_enemy_dy = 0.0
        closest_enemy_distance = 1.0
        max_possible_dist = math.hypot(self.width, self.height)
        
        closest_enemy = None
        closest_distance = float('inf')

        if enemies:
            for enemy in enemies:
                raw_dx = enemy.x - agent.x
                raw_dy = enemy.y - agent.y
                distance = math.hypot(raw_dx, raw_dy)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_enemy = enemy
                    closest_enemy_distance = min(distance / fov_range, 1.0)
                
                if distance <= fov_range:
                    enemy_count += 1
                    angle = math.atan2(raw_dy, raw_dx)
                    angle_degrees = math.degrees(angle)
                    angle_degrees = (angle_degrees + 360) % 360
                    
                    sector = int((angle_degrees + 22.5) / 45) % 8
                    
                    norm_dist = distance / fov_range
                    
                    if norm_dist < enemy_sector_distances[sector]:
                        enemy_sector_distances[sector] = norm_dist
                        vel_magnitude = math.hypot(enemy.vx, enemy.vy)
                        enemy_sector_velocities[sector] = vel_magnitude / 10.0

            enemy_count = min(1.0, enemy_count / 5.0)
            
            if closest_enemy is not None:
                raw_dx = closest_enemy.x - agent.x
                raw_dy = closest_enemy.y - agent.y
                dist = math.hypot(raw_dx, raw_dy)
                if dist > 0:
                    closest_enemy_dx = raw_dx / dist
                    closest_enemy_dy = raw_dy / dist

        # Food sensors
        if not foods:
            norm_food_dx, norm_food_dy, norm_food_distance = 0.0, 0.0, 1.0
        else:
            closest_food = min(foods, key=lambda f: math.hypot(f.x - agent.x, f.y - agent.y))
            raw_food_dx = closest_food.x - agent.x
            raw_food_dy = closest_food.y - agent.y
            min_food_dist = math.hypot(raw_food_dx, raw_food_dy)

            if min_food_dist > 0:
                norm_food_dx = raw_food_dx / min_food_dist
                norm_food_dy = raw_food_dy / min_food_dist
            else:
                norm_food_dx, norm_food_dy = 0.0, 0.0
            norm_food_distance = min_food_dist / max_possible_dist

        return NNInputs(enemy_sector_distances=enemy_sector_distances,
                        enemy_sector_velocities=enemy_sector_velocities,
                        enemy_count=enemy_count,
                        closest_enemy_dx=closest_enemy_dx,
                        closest_enemy_dy=closest_enemy_dy,
                        closest_enemy_distance=closest_enemy_distance,
                        current_energy=norm_energy,
                        current_hunger=norm_hunger,
                        food_dx=norm_food_dx,
                        food_dy=norm_food_dy,
                        food_distance=norm_food_distance,
                        prox_left=prox_left,
                        prox_right=prox_right,
                        prox_top=prox_top,
                        prox_bottom=prox_bottom
                        )