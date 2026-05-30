import neat
import pygame
import random
import math
import csv
import pickle
import os

from src.models.prey import Prey
from src.models.predator import Predator
from src.models.food import Food
from src.core.sim_controller import SimulationController
from src.core.render_controller import RenderController
import src.configs.sim_config as sim_config
from src.configs.nn_config import NNOutputs, NNInputs


def flatten_nn_inputs(inputs: NNInputs) -> list:
    """Convert NNInputs dataclass to a flat list for neural network activation."""
    flat = []
    flat.extend(inputs.enemy_sector_distances)      # 8 values
    flat.extend(inputs.enemy_sector_velocities)     # 8 values
    flat.append(inputs.enemy_count)                 # 1 value
    flat.append(inputs.closest_enemy_dx)            # 1 value
    flat.append(inputs.closest_enemy_dy)            # 1 value
    flat.append(inputs.closest_enemy_distance)      # 1 value
    flat.append(inputs.current_energy)              # 1 value
    flat.append(inputs.current_hunger)              # 1 value
    flat.append(inputs.food_dx)                     # 1 value
    flat.append(inputs.food_dy)                     # 1 value
    flat.append(inputs.food_distance)               # 1 value
    flat.append(inputs.prox_left)                   # 1 value
    flat.append(inputs.prox_right)                  # 1 value
    flat.append(inputs.prox_top)                    # 1 value
    flat.append(inputs.prox_bottom)                 # 1 value
    return flat

class AppManager:
    """Orchestrates the Co-Evolutionary NEAT algorithm and the game loop."""
    
    def __init__(
            self,
            prey_config_path: str,
            predator_config_path: str,
            width: int = sim_config.WIDTH,
            height: int = sim_config.HEIGHT,
            fps: int = sim_config.FPS,
            render_every_n: int = sim_config.RENDER_N
            ):
        
        self.width = width
        self.height = height
        self.fps = fps
        self.render_every_n = render_every_n
        
        self.render_enabled = True
        
        self.prey_config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            prey_config_path
        )

        self.predator_config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            predator_config_path
        )
        
        self.prey_pop = neat.Population(self.prey_config)
        self.predator_pop = neat.Population(self.predator_config)
        
        self.sim = SimulationController(width, height)

        self.metrics_history = []
        
        if self.render_enabled:
            self.renderer = RenderController(width, height)
        else:
            self.renderer = None
        
        self.is_paused = False
        self.selected_entity = None
        self.is_dragging = False
    
    def show_main_menu(self):
        if not self.render_enabled or not self.renderer: return "START"
            
        clock = pygame.time.Clock()
        from src.models.prey import Prey
        from src.models.predator import Predator
        from src.models.food import Food
        
        bg_sim = SimulationController(self.width, self.height)
        for _ in range(15): bg_sim.preys.append(Prey(random.randint(0, self.width), random.randint(0, self.height)))
        for _ in range(5): bg_sim.predators.append(Predator(random.randint(0, self.width), random.randint(0, self.height)))
        for _ in range(30): bg_sim.foods.append(Food(random.randint(0, self.width), random.randint(0, self.height), bg_sim.preys[0].radius / 2))

        center_x = self.renderer.screen.get_width() // 2
        center_y = self.renderer.screen.get_height() // 2
        
        start_btn = pygame.Rect(center_x - 400, center_y - 60, 200, 50)
        load_btn = pygame.Rect(center_x - 400, center_y + 10, 200, 50)
        exit_btn = pygame.Rect(center_x - 400, center_y + 80, 200, 50)
        main_buttons = [(start_btn, "Train New Models"), (load_btn, "Watch Best Models"), (exit_btn, "Exit Simulation")]

        use_custom = False
        config_state = {
            'generations': {'label': 'Generations', 'val': getattr(sim_config, 'GENERATIONS', 500), 'step': 50, 'min': 10},
            'render_n': {'label': 'Render Every N', 'val': getattr(sim_config, 'RENDER_N', 10), 'step': 1, 'min': 1},
            'max_frames': {'label': 'Max Frames', 'val': getattr(sim_config, 'MAX_FRAMES', 500), 'step': 50, 'min': 100},
            'food_count': {'label': 'Food Count', 'val': getattr(sim_config, 'FOOD_COUNT', 25), 'step': 5, 'min': 0},
            'prey_pop': {'label': 'Prey Count', 'val': self.prey_config.pop_size, 'step': 5, 'min': 5},
            'pred_pop': {'label': 'Pred Count', 'val': self.predator_config.pop_size, 'step': 1, 'min': 1},
            'prey_radius': {'label': 'Prey Radius', 'val': getattr(sim_config, 'PREY_RADIUS', 8.0), 'step': 1.0, 'min': 2.0},
            'pred_radius': {'label': 'Pred Radius', 'val': getattr(sim_config, 'PRED_RADIUS', 9.0), 'step': 1.0, 'min': 2.0},
            
            'prey_speed': {'label': 'Prey Speed', 'val': getattr(sim_config, 'PREY_SPEED', 4.0), 'step': 0.5, 'min': 0.5},
            'pred_speed': {'label': 'Pred Speed', 'val': getattr(sim_config, 'PRED_SPEED', 2.5), 'step': 0.5, 'min': 0.5},
            'prey_sprint': {'label': 'Prey Sprint', 'val': getattr(sim_config, 'PREY_SPRINT_SPEED', 8.5), 'step': 0.5, 'min': 1.0},
            'pred_sprint': {'label': 'Pred Sprint', 'val': getattr(sim_config, 'PRED_SPRINT_SPEED', 6.0), 'step': 0.5, 'min': 1.0},
            'prey_hunger': {'label': 'Prey Hunger', 'val': getattr(sim_config, 'PREY_MAX_HUNGER', 100.0), 'step': 10, 'min': 10},
            'pred_hunger': {'label': 'Pred Hunger', 'val': getattr(sim_config, 'PRED_MAX_HUNGER', 150.0), 'step': 10, 'min': 10},
            'prey_energy': {'label': 'Prey Energy', 'val': getattr(sim_config, 'PREY_MAX_ENERGY', 100.0), 'step': 10, 'min': 10},
            'pred_energy': {'label': 'Pred Energy', 'val': getattr(sim_config, 'PRED_MAX_ENERGY', 100.0), 'step': 10, 'min': 10}
        }
        
        config_rects = {}
        layout_headers = []
        panel_x = center_x - 130          
        right_col_x = panel_x + 280       
        start_y = center_y - 180
        
        config_rects['toggle'] = pygame.Rect(panel_x, start_y, 310, 30)
        
        def make_ui_row(x_pos, y_pos):
            return {
                'x': x_pos, 'y': y_pos,
                'val_box': pygame.Rect(x_pos + 115, y_pos - 2, 55, 24),
                'minus': pygame.Rect(x_pos + 180, y_pos, 20, 20),
                'plus': pygame.Rect(x_pos + 210, y_pos, 20, 20)
            }

        layout_headers.append(("System Attributes", panel_x, start_y + 40))
        config_rects['generations'] = make_ui_row(panel_x, start_y + 65)
        config_rects['max_frames'] = make_ui_row(panel_x, start_y + 95)
        config_rects['render_n'] = make_ui_row(right_col_x, start_y + 65)
        config_rects['food_count'] = make_ui_row(right_col_x, start_y + 95)
        
        layout_headers.append(("Prey Attributes", panel_x, start_y + 135))
        layout_headers.append(("Predator Attributes", right_col_x, start_y + 135))
        
        prey_keys = ['prey_pop', 'prey_radius', 'prey_speed', 'prey_sprint', 'prey_energy', 'prey_hunger']
        pred_keys = ['pred_pop', 'pred_radius', 'pred_speed', 'pred_sprint', 'pred_energy', 'pred_hunger']
        
        y_offset = start_y + 160
        for prey_k, pred_k in zip(prey_keys, pred_keys):
            config_rects[prey_k] = make_ui_row(panel_x, y_offset)
            config_rects[pred_k] = make_ui_row(right_col_x, y_offset)
            y_offset += 32

        active_input_key = None
        temp_input_text = ""

        def apply_text_input(key, text):
            if not text or text == '.' or text == '-': return
            try:
                is_float = isinstance(config_state[key]['val'], float)
                new_val = float(text) if is_float else int(text)
                config_state[key]['val'] = max(config_state[key]['min'], new_val)
            except ValueError:
                pass

        running = True
        result = "EXIT"
        
        while running:
            mx, my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                    
                if event.type == pygame.KEYDOWN and active_input_key:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                        apply_text_input(active_input_key, temp_input_text)
                        active_input_key = None
                    elif event.key == pygame.K_BACKSPACE:
                        temp_input_text = temp_input_text[:-1]
                    else:
                        char = event.unicode
                        if char.isdigit() or (char == '.' and '.' not in temp_input_text):
                            temp_input_text += char
                            
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_on_input = False
                    
                    if start_btn.collidepoint((mx, my)): running = False; result = "START"
                    if load_btn.collidepoint((mx, my)): running = False; result = "LOAD"
                    if exit_btn.collidepoint((mx, my)): pygame.quit(); exit()
                    
                    if config_rects['toggle'].collidepoint((mx, my)): use_custom = not use_custom
                    
                    if use_custom:
                        for key, rects in config_rects.items():
                            if key == 'toggle': continue
                            
                            if rects['val_box'].collidepoint((mx, my)):
                                active_input_key = key
                                val = config_state[key]['val']
                                temp_input_text = f"{val:.1f}" if isinstance(val, float) else str(int(val))
                                clicked_on_input = True
                            
                            if rects['minus'].collidepoint((mx, my)):
                                config_state[key]['val'] = max(config_state[key]['min'], config_state[key]['val'] - config_state[key]['step'])
                            if rects['plus'].collidepoint((mx, my)):
                                config_state[key]['val'] += config_state[key]['step']

                    if not clicked_on_input and active_input_key:
                        apply_text_input(active_input_key, temp_input_text)
                        active_input_key = None

            for agent in bg_sim.preys + bg_sim.predators:
                agent.x = max(0, min(self.width, agent.x + random.uniform(-2, 2)))
                agent.y = max(0, min(self.height, agent.y + random.uniform(-2, 2)))
            bg_sim.update_physics()

            self.renderer.draw_main_menu(bg_sim, mx, my, main_buttons, config_state, config_rects, use_custom, layout_headers, active_input_key, temp_input_text)
            clock.tick(30)
            
        if result == "START" and use_custom:
            sim_config.GENERATIONS = int(config_state['generations']['val'])
            sim_config.RENDER_N = int(config_state['render_n']['val'])
            sim_config.MAX_FRAMES = int(config_state['max_frames']['val'])
            sim_config.FOOD_COUNT = int(config_state['food_count']['val'])

            self.render_every_n = sim_config.RENDER_N
            
            self.prey_config.pop_size = int(config_state['prey_pop']['val'])
            self.predator_config.pop_size = int(config_state['pred_pop']['val'])
            self.prey_pop = neat.Population(self.prey_config)
            self.predator_pop = neat.Population(self.predator_config)
            
            sim_config.PREY_MAX_HUNGER = float(config_state['prey_hunger']['val'])
            sim_config.PRED_MAX_HUNGER = float(config_state['pred_hunger']['val'])
            sim_config.PREY_MAX_ENERGY = float(config_state['prey_energy']['val'])
            sim_config.PRED_MAX_ENERGY = float(config_state['pred_energy']['val'])
            
            sim_config.PREY_RADIUS = float(config_state['prey_radius']['val'])
            sim_config.PRED_RADIUS = float(config_state['pred_radius']['val'])
            sim_config.PREY_SPEED = float(config_state['prey_speed']['val'])
            sim_config.PRED_SPEED = float(config_state['pred_speed']['val'])
            sim_config.PREY_SPRINT_SPEED = float(config_state['prey_sprint']['val'])
            sim_config.PRED_SPRINT_SPEED = float(config_state['pred_sprint']['val'])
            
        return result

    def start_training(self, generations: int = sim_config.GENERATIONS):
        """The main loop. Runs the evolutionary generations."""
        print(f"Starting Co-Evolution for {generations} generations...")
        
        for generation in range(generations):
            print(f"--- Generation {generation + 1} ---")
            
            prey_genomes = list(self.prey_pop.population.items())
            predator_genomes = list(self.predator_pop.population.items())
            
            render_this_gen = self.render_enabled and ((generation + 1) % self.render_every_n == 0)
            if render_this_gen:
                print(f"Displaying Generation {generation + 1} visually...")
            
            prey_nets, predator_nets = self.evaluate_generation(prey_genomes, predator_genomes, render_this_gen)

            self._assign_scaled_fitness(prey_nets, self._calculate_prey_fitness)
            self._assign_scaled_fitness(predator_nets, self._calculate_predator_fitness)

            self.log_and_print_metrics(generation + 1, prey_nets, predator_nets)

            self._advance_generation(self.prey_pop)
            self._advance_generation(self.predator_pop)
            
        self.export_metrics_to_csv()
        
        if self.render_enabled:
            self.renderer.quit()
    
    def _advance_generation(self, pop):
        """Manually handles the NEAT reproduction and speciation lifecycle."""
        new_population = pop.reproduction.reproduce(
            pop.config, pop.species, pop.config.pop_size, pop.generation
        )

        pop.population = new_population
        pop.species.speciate(pop.config, pop.population, pop.generation)
        pop.generation += 1

    def evaluate_generation(self, prey_genomes, predator_genomes, render_this_gen=False):
        """The core Pygame loop for a single generation."""
        self.sim.preys.clear()
        self.sim.predators.clear()
        self.sim.foods.clear()
        
        prey_nets = []
        predator_nets = []
        
        center_x_min, center_x_max = self.width // 4, (self.width // 4) * 3
        center_y_min, center_y_max = self.height // 4, (self.height // 4) * 3

        for genome_id, genome in prey_genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, self.prey_config)
            
            agent = Prey(
                x=random.randint(center_x_min, center_x_max),
                y=random.randint(center_y_min, center_y_max),
                radius=sim_config.PREY_RADIUS,
                speed=sim_config.PREY_SPEED,
                sprint_speed=sim_config.PREY_SPRINT_SPEED,
                max_energy=sim_config.PREY_MAX_ENERGY,
                max_hunger=sim_config.PREY_MAX_HUNGER
            )
            
            prey_nets.append((genome, net, agent))
            self.sim.preys.append(agent)
            
        for genome_id, genome in predator_genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, self.predator_config)
            
            if random.choice([True, False]):
                px = random.choice([10, self.width - 10])
                py = random.randint(0, self.height)
            else:
                px = random.randint(0, self.width)
                py = random.choice([10, self.height - 10])
                
            agent = Predator(
                x=px,
                y=py,
                radius=sim_config.PRED_RADIUS,
                speed=sim_config.PRED_SPEED,
                sprint_speed=sim_config.PRED_SPRINT_SPEED,
                max_energy=sim_config.PRED_MAX_ENERGY,
                max_hunger=sim_config.PRED_MAX_HUNGER
            )

            predator_nets.append((genome, net, agent))
            self.sim.predators.append(agent)
        
        self.add_food()
            
        generation_frames = 0
        max_frames = sim_config.MAX_FRAMES
        is_running = True
        clock = pygame.time.Clock()
        
        current_gen_number = self.prey_pop.generation + 1
        
        while is_running and generation_frames < max_frames:
            if self.render_enabled:
                self._handle_events()
            
            if not self.is_paused:
                missing_food = sim_config.FOOD_COUNT - len(self.sim.foods)
                if missing_food > 0 and generation_frames % sim_config.FOOD_GENERATION_FRAMES == 0:
                    self.add_food(missing_food)
                
                # INTENTION (Move and Record State)
                for genome, net, agent in prey_nets:
                    if agent.is_alive:
                        agent.foods_before_step = agent.foods
                        agent.last_predator_distance = self._closest_predator_distance(agent)
                        agent.last_food_distance = self._closest_food_distance(agent)

                        inputs = self.sim.get_agent_inputs(agent, self.sim.predators, self.sim.foods)
                        outputs = net.activate(flatten_nn_inputs(inputs))
                        agent.apply_action(NNOutputs(*outputs), self.width, self.height)

                for genome, net, agent in predator_nets:
                    if agent.is_alive:
                        agent.kills_before_step = agent.kills
                        inputs = self.sim.get_agent_inputs(agent, self.sim.preys, self.sim.foods)
                        agent.last_prey_distance = self._closest_prey_distance(agent)
                        outputs = net.activate(flatten_nn_inputs(inputs))
                        agent.apply_action(NNOutputs(*outputs), self.width, self.height)
                        if inputs.enemy_count > 0:
                            agent.seen_prey_frames += 1
                            agent.hunt_score += 1.0 - inputs.closest_enemy_distance

                        if agent.attack_cooldown > 0:
                            agent.attack_cooldown = max(0, agent.attack_cooldown - 1)

                # RESOLUTION (Physics & Collisions)
                self.sim.update_physics()
                
                for genome, net, agent in predator_nets:
                    if agent.is_alive and agent.last_prey_distance is not None:
                        new_dist = self._closest_prey_distance(agent)
                        if new_dist is not None:
                            agent.approach_score += max(0.0, agent.last_prey_distance - new_dist)

                for genome, net, agent in prey_nets:
                    if not agent.is_alive:
                        continue

                    if getattr(agent, "last_food_distance", None) is not None:
                        new_food_dist = self._closest_food_distance(agent)
                        if new_food_dist is not None:
                            agent.food_approach_score = getattr(agent, "food_approach_score", 0.0)
                            agent.food_approach_score += max(0.0, agent.last_food_distance - new_food_dist)

                    if getattr(agent, "last_predator_distance", None) is not None:
                        new_pred_dist = self._closest_predator_distance(agent)
                        if new_pred_dist is not None:
                            if agent.last_predator_distance < sim_config.PREY_DANGER_RADIUS:
                                agent.danger_frames = getattr(agent, "danger_frames", 0) + 1
                                agent.escape_score = getattr(agent, "escape_score", 0.0)
                                agent.escape_score += max(0.0, new_pred_dist - agent.last_predator_distance)
                
                # CONSEQUENCE (Hunger & Survival)
                for genome, net, agent in prey_nets:
                    if agent.is_alive:
                        if agent.foods > agent.foods_before_step:
                            agent.current_hunger = agent.max_hunger
                        else:
                            agent.current_hunger -= agent.hunger_consumption
                        
                        if agent.current_hunger <= 0.0:
                            agent.is_alive = False
                        else:
                            agent.survival_frames += 1

                for genome, net, agent in predator_nets:
                    if agent.is_alive:
                        if agent.kills > agent.kills_before_step:
                            agent.current_hunger = agent.max_hunger
                        else:
                            agent.current_hunger -= agent.hunger_consumption
                            
                        if agent.current_hunger <= 0.0:
                            agent.is_alive = False
                        else:
                            agent.survival_frames += 1
                
                generation_frames += 1

            if render_this_gen:
                alive_prey = len([p for p in self.sim.preys if p.is_alive])
                alive_preds = len([p for p in self.sim.predators if p.is_alive])
                
                live_stats = {
                    "Generation": current_gen_number,
                    "Frame": f"{generation_frames} / {max_frames}",
                    "Alive Prey": alive_prey,
                    "Alive Preds": alive_preds,
                    "Total Food": len(self.sim.foods)
                }
                
                self.renderer.draw_frame(
                    self.sim, 
                    stats=live_stats, 
                    selected_entity=self.selected_entity, 
                    is_paused=self.is_paused
                )
                clock.tick(self.fps)
            else:
                if generation_frames % 30 == 0:
                    self.renderer.draw_training_screen(current_gen_number, sim_config.GENERATIONS, self.is_paused)
                
            if len(self.sim.preys) == 0:
                is_running = False

        return prey_nets, predator_nets

    def _calculate_prey_fitness(self, agent, genome) -> float:
        node_cost = len(genome.nodes) * 2
        conn_cost = len([c for c in genome.connections.values() if c.enabled]) * 0.5
        metabolic_penalty = node_cost + conn_cost
        
        alive_bonus = 400 if agent.is_alive else 0

        food_approach_score = getattr(agent, "food_approach_score", 0.0)
        escape_score = getattr(agent, "escape_score", 0.0)
        danger_frames = getattr(agent, "danger_frames", 0)
        contact_danger_frames = getattr(agent, "contact_danger_frames", 0)
        idle_frames = getattr(agent, "idle_frames", 0)
        jitter_frames = getattr(agent, "jitter_frames", 0)

        fitness = (
            (agent.foods * 900.0)
            + (food_approach_score * 8.0)
            + (escape_score * 16.0)
            + (agent.survival_frames * 2.0)
            + alive_bonus
            - (danger_frames * 1.0)
            - (contact_danger_frames * 60.0)
            - (agent.wall_frames * 25.0)
            - (idle_frames * 1.0)
            - (jitter_frames * 0.5)
            - metabolic_penalty
        )

        return fitness

    def _calculate_predator_fitness(self, agent, genome) -> float:
        node_cost = len(genome.nodes) * 2
        conn_cost = len([c for c in genome.connections.values() if c.enabled]) * 0.5
        metabolic_penalty = node_cost + conn_cost

        first_kill = min(agent.kills, 1)
        second_kill = min(max(agent.kills - 1, 0), 1)
        extra_kills = max(agent.kills - 2, 0)

        kill_score = (
            first_kill * 2200.0
            + second_kill * 900.0
            + extra_kills * 300.0
        )

        camp_frames = getattr(agent, "camp_frames", 0)
        idle_frames = getattr(agent, "idle_frames", 0)
        jitter_frames = getattr(agent, "jitter_frames", 0)

        fitness = (
            kill_score
            + (agent.approach_score * 10.0)
            + (agent.hunt_score * 0.2)
            + (agent.seen_prey_frames * 0.02)
            - (camp_frames * 35.0)
            - (idle_frames * 2.0)
            - (jitter_frames * 3.0)
            - (agent.wall_frames * 20.0)
            - metabolic_penalty
        )

        return fitness
    
    def _assign_scaled_fitness(self, nets, raw_fn):
        raws = [raw_fn(agent, genome) for genome, _, agent in nets]
        min_raw = min(raws)

        for (genome, _, _), raw in zip(nets, raws):
            genome.fitness = max(0.001, raw - min_raw + 1.0)
    
    def _closest_prey_distance(self, predator):
        living_preys = [p for p in self.sim.preys if p.is_alive]
        if not living_preys:
            return None
        return min(math.hypot(p.x - predator.x, p.y - predator.y) for p in living_preys)
    
    def _closest_predator_distance(self, prey):
        living_predators = [p for p in self.sim.predators if p.is_alive]
        if not living_predators:
            return None
        return min(math.hypot(p.x - prey.x, p.y - prey.y) for p in living_predators)
    
    def _closest_food_distance(self, prey):
        if not self.sim.foods:
            return None
        return min(math.hypot(f.x - prey.x, f.y - prey.y) for f in self.sim.foods)
    
    def add_food(self, n: int | None = None):
        if n is None:
            n = sim_config.FOOD_COUNT

        for _ in range(n):
            food = Food(
                random.randint(0, self.width),
                random.randint(0, self.height),
                self.sim.preys[0].radius / 2
            )
            self.sim.foods.append(food)

    def _handle_events(self):
        """Listens for Pygame events: quit signals, UI interactions, and hotkeys."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            
            # --- KEYBOARD CONTROLS ---
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                
                if self.selected_entity:
                    # Kill / Delete
                    if event.key in [pygame.K_DELETE, pygame.K_BACKSPACE]:
                        if hasattr(self.selected_entity, 'is_alive'):
                            self.selected_entity.is_alive = False
                        elif self.selected_entity in self.sim.foods:
                            self.sim.foods.remove(self.selected_entity)
                        self.selected_entity = None
                        self.is_dragging = False
                        
                    # Refill Hunger
                    elif event.key == pygame.K_h and hasattr(self.selected_entity, 'current_hunger'):
                        self.selected_entity.current_hunger = self.selected_entity.max_hunger
                        
                    # Refill Energy
                    elif event.key == pygame.K_e and hasattr(self.selected_entity, 'current_energy'):
                        self.selected_entity.current_energy = self.selected_entity.max_energy

            # --- MOUSE CONTROLS ---
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if mx < self.width:
                        closest = None
                        min_dist = float('inf')
                        
                        all_entities = self.sim.preys + self.sim.predators + self.sim.foods
                        
                        for entity in all_entities:
                            if hasattr(entity, 'is_alive') and not entity.is_alive:
                                continue
                                
                            dist = math.hypot(entity.x - mx, entity.y - my)
                            if dist < (entity.radius + 5) and dist < min_dist:
                                closest = entity
                                min_dist = dist
                                
                        self.selected_entity = closest
                        if self.selected_entity:
                            self.is_dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.is_dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.is_dragging and self.selected_entity:
                    mx, my = event.pos
                    if 0 < mx < self.width and 0 < my < self.height:
                        self.selected_entity.x = mx
                        self.selected_entity.y = my

    def log_and_print_metrics(self, generation, prey_nets, predator_nets):
        """Extracts behavioral data, prints a summary, and stores it for graphing."""
        
        prey_survivals = [agent.survival_frames for _, _, agent in prey_nets]
        prey_foods = [agent.foods for _, _, agent in prey_nets]
        
        avg_prey_survival = sum(prey_survivals) / len(prey_survivals) if prey_survivals else 0
        max_prey_survival = max(prey_survivals) if prey_survivals else 0
        total_food = sum(prey_foods)

        predator_kills = [agent.kills for _, _, agent in predator_nets]
        predator_survivals = [agent.survival_frames for _, _, agent in predator_nets]
        
        avg_predator_kills = sum(predator_kills) / len(predator_kills) if predator_kills else 0
        max_predator_kills = max(predator_kills) if predator_kills else 0
        total_kills = sum(predator_kills)

        prey_nodes = [len(genome.nodes) for genome, _, _ in prey_nets]
        prey_conns = [len([c for c in genome.connections.values() if c.enabled]) for genome, _, _ in prey_nets]
        pred_nodes = [len(genome.nodes) for genome, _, _ in predator_nets]
        pred_conns = [len([c for c in genome.connections.values() if c.enabled]) for genome, _, _ in predator_nets]

        alive_prey_end = sum(1 for _, _, agent in prey_nets if agent.is_alive)
        alive_pred_end = sum(1 for _, _, agent in predator_nets if agent.is_alive)

        stats = {
            "generation": generation,
            "avg_prey_survival": round(avg_prey_survival, 2),
            "max_prey_survival": max_prey_survival,
            "total_food": total_food,
            "avg_predator_kills": round(avg_predator_kills, 2),
            "max_predator_kills": max_predator_kills,
            "total_kills": total_kills,
            "avg_prey_nodes": round(sum(prey_nodes) / len(prey_nodes), 2),
            "avg_prey_connections": round(sum(prey_conns) / len(prey_conns), 2),
            "avg_pred_nodes": round(sum(pred_nodes) / len(pred_nodes), 2),
            "avg_pred_connections": round(sum(pred_conns) / len(pred_conns), 2),
            "alive_prey_end": alive_prey_end,
            "alive_pred_end": alive_pred_end
        }
        self.metrics_history.append(stats)

        print(f"PREY     | Avg Survival: {stats['avg_prey_survival']:<6} | Max Survival: {stats['max_prey_survival']:<4} | Total Food: {stats['total_food']}")
        print(f"PREDATOR | Avg Kills:    {stats['avg_predator_kills']:<6} | Max Kills:    {stats['max_predator_kills']:<4} | Total Kills: {stats['total_kills']}")
        print("-" * 60)

    def export_metrics_to_csv(self, filename="results/evolution_metrics.csv"):
        """Saves the history to a CSV for your project report graphs."""
        if not self.metrics_history:
            return
            
        keys = self.metrics_history[0].keys()
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.metrics_history)
        print(f"Metrics successfully exported to {filename}")

    def save_best_models(self, prey_genome, pred_genome, filename="best_models.pkl"):
        """Saves the highest performing genomes to a file."""
        os.makedirs("checkpoints", exist_ok=True)
        filepath = os.path.join("checkpoints", filename)
        with open(filepath, 'wb') as f:
            pickle.dump({'prey': prey_genome, 'predator': pred_genome}, f)
        print(f"SUCCESS: Best models saved to {filepath}")

    def load_best_models(self, filename="best_models.pkl"):
        """Loads the best genomes from a file."""
        filepath = os.path.join("checkpoints", filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"SUCCESS: Models loaded from {filepath}")
            return data['prey'], data['predator']
        else:
            print(f"ERROR: Could not find {filepath}")
            return None, None