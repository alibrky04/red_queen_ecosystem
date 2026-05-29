import pygame
import math
from src.core.sim_controller import SimulationController
import src.configs.sim_config as sim_config

class RenderController:
    """Handles all Pygame window initialization, drawing, and UI."""
    
    def __init__(self, width: int, height: int):
        pygame.init()
        pygame.font.init()
        
        self.sim_width = width
        self.sim_height = height
        self.sidebar_width = getattr(sim_config, 'SIDEBAR_WIDTH', 250)
        
        self.screen = pygame.display.set_mode((self.sim_width + self.sidebar_width, height))
        pygame.display.set_caption("Red Queen Co-Evolution MVP - Interactive HUD")
        
        # Fonts
        self.font = pygame.font.SysFont("Consolas", 14)
        self.title_font = pygame.font.SysFont("Consolas", 16, bold=True)
        
        # Colors
        self.BG_COLOR = (30, 30, 30)
        self.SIDEBAR_COLOR = (20, 20, 20)
        self.TEXT_COLOR = (220, 220, 220)
        self.PREY_COLOR = (100, 255, 100)
        self.PREDATOR_COLOR = (255, 100, 100)
        self.FOOD_COLOR = (100, 100, 255)
        self.SELECT_COLOR = (255, 255, 0)
    
    def draw_main_menu(self, bg_sim, mouse_x, mouse_y, main_buttons, config_state, config_rects, use_custom, layout_headers, active_input_key=None, temp_input_text=""):
        self.draw_frame(bg_sim, do_flip=False)
        
        overlay = pygame.Surface((self.screen.get_width(), self.screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 215))
        self.screen.blit(overlay, (0, 0))

        center_x = self.screen.get_width() // 2
        center_y = self.screen.get_height() // 2

        title = self.title_font.render("RED QUEEN CO-EVOLUTION", True, (255, 255, 255))
        self.screen.blit(title, (center_x - 400 + 100 - title.get_width()//2, center_y - 150))

        for btn_rect, text in main_buttons:
            color = (80, 150, 200) if btn_rect.collidepoint((mouse_x, mouse_y)) else (50, 50, 50)
            pygame.draw.rect(self.screen, color, btn_rect, border_radius=5)
            pygame.draw.rect(self.screen, (200, 200, 200), btn_rect, 2, border_radius=5)
            
            label = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(label, (btn_rect.x + btn_rect.width//2 - label.get_width()//2, btn_rect.y + btn_rect.height//2 - label.get_height()//2))

        toggle_rect = config_rects['toggle']
        toggle_color = self.PREY_COLOR if use_custom else (80, 80, 80)
        pygame.draw.rect(self.screen, toggle_color, toggle_rect, border_radius=4)
        t_text = self.font.render("Custom Settings: ON" if use_custom else "Custom Settings: OFF (Using Defaults)", True, (0,0,0))
        self.screen.blit(t_text, (toggle_rect.x + 10, toggle_rect.y + 7))

        for text, x, y in layout_headers:
            header_color = self.TEXT_COLOR if use_custom else (100, 100, 100)
            lbl = self.title_font.render(text, True, header_color)
            self.screen.blit(lbl, (x, y))
            if use_custom:
                pygame.draw.line(self.screen, (80, 80, 80), (x, y + 18), (x + 230, y + 18))

        for key, data in config_state.items():
            rects = config_rects[key]
            x_pos, y_pos = rects['x'], rects['y']
            val_rect = rects['val_box']
            
            lbl = self.font.render(data['label'], True, (200, 200, 200))
            self.screen.blit(lbl, (x_pos, y_pos + 2))
            
            if use_custom:
                box_color = (60, 60, 60) if val_rect.collidepoint((mouse_x, mouse_y)) or key == active_input_key else (30, 30, 30)
                pygame.draw.rect(self.screen, box_color, val_rect, border_radius=3)
            
            if key == active_input_key:
                cursor = "|" if (pygame.time.get_ticks() // 500) % 2 == 0 else ""
                val_text = self.title_font.render(temp_input_text + cursor, True, (255, 255, 255))
            else:
                val_color = self.SELECT_COLOR if use_custom else (100, 100, 100)
                val_str = f"{data['val']:.1f}" if isinstance(data['val'], float) else str(int(data['val']))
                val_text = self.title_font.render(val_str, True, val_color)
            
            self.screen.blit(val_text, (val_rect.x + val_rect.width//2 - val_text.get_width()//2, y_pos + 2))
            
            if use_custom:
                m_rect, p_rect = rects['minus'], rects['plus']
                c_minus = (180, 80, 80) if m_rect.collidepoint((mouse_x, mouse_y)) else (80, 80, 80)
                c_plus = (80, 180, 80) if p_rect.collidepoint((mouse_x, mouse_y)) else (80, 80, 80)
                
                pygame.draw.rect(self.screen, c_minus, m_rect, border_radius=3)
                pygame.draw.rect(self.screen, c_plus, p_rect, border_radius=3)
                self.screen.blit(self.title_font.render("-", True, (255,255,255)), (m_rect.x + 6, m_rect.y + 1))
                self.screen.blit(self.title_font.render("+", True, (255,255,255)), (p_rect.x + 5, p_rect.y + 1))

        pygame.display.flip()

    def draw_training_screen(self, current_gen, max_gen, is_paused=False):
        """Draws a cute chase animation while the simulation solves physics in the background."""
        self.screen.fill(self.BG_COLOR)
        
        center_x = self.screen.get_width() // 2
        center_y = self.screen.get_height() // 2
        
        t = pygame.time.get_ticks() / 350.0  
        radius = 60
        
        prey_x = center_x + math.cos(t) * radius
        prey_y = center_y - 50 + math.sin(t) * radius
        
        pred_x = center_x + math.cos(t - 0.5) * radius
        pred_y = center_y - 50 + math.sin(t - 0.5) * radius
        
        pygame.draw.circle(self.screen, self.PREY_COLOR, (int(prey_x), int(prey_y)), 8)
        pygame.draw.circle(self.screen, self.PREDATOR_COLOR, (int(pred_x), int(pred_y)), 10)
        
        status = "|| PAUSED ||" if is_paused else "TRAINING IN PROGRESS..."
        title_color = (255, 100, 100) if is_paused else self.TEXT_COLOR
        title = self.title_font.render(status, True, title_color)
        self.screen.blit(title, (center_x - title.get_width()//2, center_y + 40))
        
        gen_text = self.font.render(f"Generation: {current_gen} / {max_gen}", True, self.TEXT_COLOR)
        self.screen.blit(gen_text, (center_x - gen_text.get_width()//2, center_y + 70))
        
        pygame.display.flip()

    def draw_frame(self, sim, stats=None, selected_entity=None, is_paused=False, do_flip=True):
        self.screen.fill(self.BG_COLOR, (0, 0, self.sim_width, self.sim_height))
        self.screen.fill(self.SIDEBAR_COLOR, (self.sim_width, 0, self.sidebar_width, self.sim_height))

        for food in sim.foods:
            pygame.draw.circle(self.screen, self.FOOD_COLOR, (int(food.x), int(food.y)), int(food.radius))

        for prey in sim.preys:
            pygame.draw.circle(self.screen, self.PREY_COLOR, (int(prey.x), int(prey.y)), int(prey.radius))
            bar_w = int((prey.current_hunger / prey.max_hunger) * 20)
            pygame.draw.rect(self.screen, (80, 80, 80), (int(prey.x) - 10, int(prey.y) - 14, 20, 3))
            pygame.draw.rect(self.screen, (80, 220, 80), (int(prey.x) - 10, int(prey.y) - 14, bar_w, 3))

        for predator in sim.predators:
            r = min(255, 150 + predator.kills * 20)
            pygame.draw.circle(self.screen, (r, 80, 80), (int(predator.x), int(predator.y)), int(predator.radius))
            bar_w = int((predator.current_hunger / predator.max_hunger) * 24)
            pygame.draw.rect(self.screen, (80, 80, 80), (int(predator.x) - 12, int(predator.y) - 16, 24, 3))
            pygame.draw.rect(self.screen, (220, 80, 80), (int(predator.x) - 12, int(predator.y) - 16, bar_w, 3))

        if selected_entity:
            is_alive = getattr(selected_entity, 'is_alive', True)
            if is_alive:
                pygame.draw.circle(self.screen, self.SELECT_COLOR, (int(selected_entity.x), int(selected_entity.y)), int(selected_entity.radius) + 3, 2)

        y_offset = 15
        x_offset = self.sim_width + 15
        
        def render_text(text, color, bold=False):
            nonlocal y_offset
            f = self.title_font if bold else self.font
            surface = f.render(text, True, color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 20
        
        # --- Stats Section ---
        render_text("--- LIVE METRICS ---", self.TEXT_COLOR, bold=True)
        if stats:
            for k, v in stats.items():
                render_text(f"{k}: {v}", self.TEXT_COLOR)
        
        y_offset += 15
        # --- Controls Section ---
        render_text("--- CONTROLS ---", self.TEXT_COLOR, bold=True)
        render_text("[SPACE] Play/Pause", self.TEXT_COLOR)
        render_text("[CLICK] Select Agent/Food", self.TEXT_COLOR)
        render_text("[DRAG]  Move Selected", self.TEXT_COLOR)
        render_text("[DEL]   Kill/Remove", self.TEXT_COLOR)
        render_text("[H]     Refill Hunger", self.TEXT_COLOR)
        render_text("[E]     Refill Energy", self.TEXT_COLOR)
        
        if is_paused:
            y_offset += 10
            render_text("|| SIMULATION PAUSED ||", (255, 100, 100), bold=True)

        y_offset += 15
        # --- Selected Entity Info ---
        if selected_entity and getattr(selected_entity, 'is_alive', True):
            render_text("--- INSPECTOR ---", self.SELECT_COLOR, bold=True)
            render_text(f"Type: {type(selected_entity).__name__}", self.TEXT_COLOR)
            
            if hasattr(selected_entity, 'current_energy'):
                render_text(f"Energy: {selected_entity.current_energy:.1f}/{selected_entity.max_energy}", self.TEXT_COLOR)
            if hasattr(selected_entity, 'current_hunger'):
                render_text(f"Hunger: {selected_entity.current_hunger:.1f}/{selected_entity.max_hunger}", self.TEXT_COLOR)
            if hasattr(selected_entity, 'kills'):
                render_text(f"Kills: {selected_entity.kills}", self.TEXT_COLOR)
            if hasattr(selected_entity, 'foods'):
                render_text(f"Foods: {selected_entity.foods}", self.TEXT_COLOR)
            if hasattr(selected_entity, 'survival_frames'):
                render_text(f"Age (frames): {selected_entity.survival_frames}", self.TEXT_COLOR)

        if do_flip:
            pygame.display.flip()

    def quit(self):
        pygame.quit()