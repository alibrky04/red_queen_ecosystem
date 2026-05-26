from src.models.base_agent import BaseAgent

class Predator(BaseAgent):
    """Predator agent. Slower, larger radius. Hunts prey."""
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y, speed=3.5, sprint_speed=9.0, radius=12.0, max_energy=80.0)