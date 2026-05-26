import pygame
from src.core.sim_controller import SimulationController

class RenderController:
    """Handles all Pygame window initialization and drawing."""
    
    def __init__(self, width: int, height: int):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Red Queen Co-Evolution MVP")
        
        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.PREY_COLOR = (100, 255, 100)      # Green
        self.PREDATOR_COLOR = (255, 100, 100)  # Red
        self.FOOD_Color = (100, 100, 255)      # Blue

    def draw_frame(self, sim: "SimulationController"):
        """Draws the current state of the simulation."""
        self.screen.fill(self.BG_COLOR)
        
        # Draw Foods
        for food in sim.foods:
            pygame.draw.circle(self.screen, self.FOOD_Color, (int(food.x), int(food.y)), int(food.radius))

        # Draw Prey
        for prey in sim.preys:
            pygame.draw.circle(self.screen, self.PREY_COLOR, (int(prey.x), int(prey.y)), int(prey.radius))
            
        # Draw Predators
        for predator in sim.predators:
            pygame.draw.circle(self.screen, self.PREDATOR_COLOR, (int(predator.x), int(predator.y)), int(predator.radius))
            
        pygame.display.flip()

    def quit(self):
        pygame.quit()