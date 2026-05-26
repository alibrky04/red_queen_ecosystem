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
        """Calculates normalized sensory inputs for a given agent."""
        
        whisker_len = 200.0
        prox_left = max(0.0, 1.0 - (agent.x / whisker_len))
        prox_right = max(0.0, 1.0 - ((self.width - agent.x) / whisker_len))
        prox_top = max(0.0, 1.0 - (agent.y / whisker_len))
        prox_bottom = max(0.0, 1.0 - ((self.height - agent.y) / whisker_len))

        norm_energy = max(0, agent.current_energy) / agent.max_energy
        norm_hunger = max(0, agent.current_hunger) / agent.max_hunger

        # Enemy sensors
        if not enemies:
            norm_dx, norm_dy, norm_distance = 0.0, 0.0, 1.0
            norm_enemy_vel_dx, norm_enemy_vel_dy = 0.0, 0.0
        else:
            closest_enemy = min(enemies, key=lambda e: math.hypot(e.x - agent.x, e.y - agent.y))
            raw_dx = closest_enemy.x - agent.x
            raw_dy = closest_enemy.y - agent.y
            min_enemy_dist = math.hypot(raw_dx, raw_dy)
            max_possible_dist = math.hypot(self.width, self.height)

            if min_enemy_dist > 0:
                norm_dx = raw_dx / min_enemy_dist
                norm_dy = raw_dy / min_enemy_dist
            else:
                norm_dx, norm_dy = 0.0, 0.0
            norm_distance = min_enemy_dist / max_possible_dist

            norm_enemy_vel_dx = closest_enemy.vx
            norm_enemy_vel_dy = closest_enemy.vy

        # Food sensors
        if not foods:
            norm_food_dx, norm_food_dy, norm_food_distance = 0.0, 0.0, 1.0
        else:
            closest_food = min(foods, key=lambda f: math.hypot(f.x - agent.x, f.y - agent.y))
            raw_food_dx = closest_food.x - agent.x
            raw_food_dy = closest_food.y - agent.y
            min_food_dist = math.hypot(raw_food_dx, raw_food_dy)
            max_possible_dist = math.hypot(self.width, self.height)

            if min_food_dist > 0:
                norm_food_dx = raw_food_dx / min_food_dist
                norm_food_dy = raw_food_dy / min_food_dist
            else:
                norm_food_dx, norm_food_dy = 0.0, 0.0
            norm_food_distance = min_food_dist / max_possible_dist

        return NNInputs(enemy_dx=norm_dx,
                        enemy_dy=norm_dy,
                        enemy_distance=norm_distance,
                        enemy_vel_dx=norm_enemy_vel_dx,
                        enemy_vel_dy=norm_enemy_vel_dy,
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