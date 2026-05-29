from src.models.base_agent import BaseAgent
import src.configs.sim_config as sim_config

class Prey(BaseAgent):
    def __init__(
            self,
            x: float,
            y: float,
            radius: float = sim_config.PREY_RADIUS,
            speed: float = sim_config.PREY_SPEED,
            sprint_speed: float = sim_config.PREY_SPRINT_SPEED,
            max_energy: float = sim_config.PREY_MAX_ENERGY,
            max_hunger: float = sim_config.PREY_MAX_HUNGER
        ):
        
        super().__init__(
            x=x,
            y=y,
            radius=radius,
            speed=speed,
            sprint_speed=sprint_speed,
            max_energy=max_energy,
            max_hunger=max_hunger
        )