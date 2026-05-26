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
        
        whisker_len = 100.0
        prox_left = max(0.0, 1.0 - (agent.x / whisker_len))
        prox_right = max(0.0, 1.0 - ((self.width - agent.x) / whisker_len))
        prox_top = max(0.0, 1.0 - (agent.y / whisker_len))
        prox_bottom = max(0.0, 1.0 - ((self.height - agent.y) / whisker_len))

        norm_energy = max(0, agent.current_energy) / agent.max_energy
        norm_hunger = max(0, agent.current_hunger) / agent.max_hunger

        # --- Enemy Sensors ---
        if not enemies:
            norm_dx, norm_dy, norm_distance = 0.0, 0.0, 1.0
        else:
            closest_enemy = min(enemies, key=lambda e: math.hypot(e.x - agent.x, e.y - agent.y))
            min_enemy_dist = math.hypot(closest_enemy.x - agent.x, closest_enemy.y - agent.y)
            max_possible_dist = math.hypot(self.width, self.height)
            
            norm_dx = (closest_enemy.x - agent.x) / self.width
            norm_dy = (closest_enemy.y - agent.y) / self.height
            norm_distance = min_enemy_dist / max_possible_dist

        # --- Food Sensors ---
        if not foods:
            norm_food_dx, norm_food_dy, norm_food_distance = 0.0, 0.0, 1.0
        else:
            closest_food = min(foods, key=lambda f: math.hypot(f.x - agent.x, f.y - agent.y))
            min_food_dist = math.hypot(closest_food.x - agent.x, closest_food.y - agent.y)
            max_possible_dist = math.hypot(self.width, self.height)
            
            norm_food_dx = (closest_food.x - agent.x) / self.width
            norm_food_dy = (closest_food.y - agent.y) / self.height
            norm_food_distance = min_food_dist / max_possible_dist

        return NNInputs(norm_dx, norm_dy, norm_distance, norm_energy, norm_hunger,
                        norm_food_dx, norm_food_dy, norm_food_distance,
                        prox_left, prox_right, prox_top, prox_bottom)