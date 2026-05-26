from src.models.base_agent import BaseAgent

class Prey(BaseAgent):
    """Prey agent. Faster, smaller radius."""
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y, speed=4.0, sprint_speed=7.0, radius=8.0, max_energy=200.0)