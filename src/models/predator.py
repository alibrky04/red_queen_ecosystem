from src.models.base_agent import BaseAgent

class Predator(BaseAgent):
    """Predator agent. Slower, larger radius. Hunts prey."""
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y, speed=2.0, sprint_speed=3.0, radius=8.0, max_energy=100.0)
        self.kills = 0