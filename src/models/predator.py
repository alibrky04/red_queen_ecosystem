from src.models.base_agent import BaseAgent
import src.configs.sim_config as sim_config

class Predator(BaseAgent):
    def __init__(
		self,
		x: float,
		y: float,
		radius: float = sim_config.PRED_RADIUS,
		speed: float = sim_config.PRED_SPEED,
		sprint_speed: float = sim_config.PRED_SPRINT_SPEED,
		max_energy: float = sim_config.PRED_MAX_ENERGY,
		max_hunger: float = sim_config.PRED_MAX_HUNGER
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

        self.hunt_score = 0.0
        self.approach_score = 0.0
        self.seen_prey_frames = 0
        self.last_prey_distance = None

        self.attack_cooldown = 0
        self.camp_frames = 0

        self.hunger_consumption = sim_config.PRED_HUNGER_CONS