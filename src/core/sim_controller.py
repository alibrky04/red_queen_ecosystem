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

    def update_physics(self):
        self._check_collisions()

    def _check_collisions(self):
        """Calculates distance between predators and prey to determine eating."""
        for predator in self.predators:
            for prey in self.preys:
                if not prey.is_alive:
                    continue
                
                # Circular collision math
                dist = math.hypot(predator.x - prey.x, predator.y - prey.y)
                if dist < (predator.radius + prey.radius):
                    prey.is_alive = False
                    predator.kills += 1
                    
        self.preys = [p for p in self.preys if p.is_alive]
    
    def get_agent_inputs(self, agent, enemies):
        """Calculates normalized sensory inputs for a given agent."""
        
        dist_to_left = agent.x / self.width
        dist_to_right = (self.width - agent.x) / self.width
        dist_to_top = agent.y / self.height
        dist_to_bottom = (self.height - agent.y) / self.height

        if not enemies:
            return NNInputs(0.0, 0.0, 1.0, agent.current_energy / agent.max_energy,
                          dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

        closest_enemy = None
        min_dist = float('inf')
        
        for enemy in enemies:
            dist = math.hypot(enemy.x - agent.x, enemy.y - agent.y)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy

        agent.closest_enemy_distance = min(agent.closest_enemy_distance, min_dist)

        raw_dx = closest_enemy.x - agent.x
        raw_dy = closest_enemy.y - agent.y

        norm_dx = raw_dx / self.width
        norm_dy = raw_dy / self.height
        
        max_possible_dist = math.hypot(self.width, self.height)
        norm_distance = min_dist / max_possible_dist 
        
        norm_energy = max(0, agent.current_energy) / agent.max_energy

        return NNInputs(norm_dx, norm_dy, norm_distance, norm_energy,
                       dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)