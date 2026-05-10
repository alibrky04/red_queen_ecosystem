import neat
import pygame
import random
import math
from dataclasses import astuple

from src.models.prey import Prey
from src.models.predator import Predator
from src.core.sim_controller import SimulationController
from src.core.render_controller import RenderController
import src.configs.sim_config as sim_config
from src.configs.nn_config import NNOutputs

class AppManager:
    """Orchestrates the Co-Evolutionary NEAT algorithm and the game loop."""
    
    def __init__(
            self,
            config_path: str,
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
        
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path
        )
        
        self.prey_pop = neat.Population(self.config)
        self.predator_pop = neat.Population(self.config)
        
        self.sim = SimulationController(width, height)
        
        if self.render_enabled:
            self.renderer = RenderController(width, height)
        else:
            self.renderer = None

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
            
            self.evaluate_generation(prey_genomes, predator_genomes, render_this_gen)
            
            self._advance_generation(self.prey_pop)
            self._advance_generation(self.predator_pop)
            
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
        
        prey_nets = []
        predator_nets = []
        
        for genome_id, genome in prey_genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            agent = Prey(random.randint(0, self.width), random.randint(0, self.height))
            prey_nets.append((genome, net, agent))
            self.sim.preys.append(agent)
            
        for genome_id, genome in predator_genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            agent = Predator(random.randint(0, self.width), random.randint(0, self.height))
            predator_nets.append((genome, net, agent))
            self.sim.predators.append(agent)
            
        generation_frames = 0
        max_frames = sim_config.MAX_FRAMES
        is_running = True
        clock = pygame.time.Clock()
        
        while is_running and generation_frames < max_frames:
            if self.render_enabled:
                self._handle_events()
            
            for genome, net, agent in prey_nets:
                if agent.is_alive:
                    a = {}
                    inputs = self.sim.get_agent_inputs(agent, self.sim.predators)
                    outputs = net.activate(list(astuple(inputs)))
                    agent.apply_action(NNOutputs(*outputs), self.width, self.height)
                    agent.survival_frames += 1
            
            for genome, net, agent in predator_nets:
                if agent.is_alive:
                    inputs = self.sim.get_agent_inputs(agent, self.sim.preys)
                    outputs = net.activate(list(astuple(inputs)))
                    agent.apply_action(NNOutputs(*outputs), self.width, self.height)
                    agent.survival_frames += 1
            
            self.sim.update_physics()
            
            if render_this_gen:
                self.renderer.draw_frame(self.sim)
                clock.tick(self.fps)
                
            generation_frames += 1
            
            if len(self.sim.preys) == 0:
                is_running = False

        # Evaluate fitness for each species
        for genome, net, agent in prey_nets:
            genome.fitness = self._calculate_prey_fitness(agent)
            
        for genome, net, agent in predator_nets:
            genome.fitness = self._calculate_predator_fitness(agent)

    def _calculate_prey_fitness(self, agent) -> float:
        """Calculate fitness for a prey agent.
        
        Components:
        - Survival: how long did they last (primary goal)
        - Movement: distance traveled (discourages corner camping)
        - Corner penalty: punishment for being near edges
        """
        survival_bonus = agent.survival_frames
        movement_bonus = agent.total_distance * 0.5
        
        min_wall_dist = min(
            agent.x,
            agent.y,
            self.width - agent.x,
            self.height - agent.y
        )
        corner_penalty = 0 if min_wall_dist > 50 else (50 - min_wall_dist) * 2
        
        return survival_bonus + movement_bonus - corner_penalty

    def _calculate_predator_fitness(self, agent) -> float:
        """Calculate fitness for a predator agent.
        
        Components:
        - Kill reward: successful hunts (primary goal)
        - Proximity bonus: getting close to prey (guides early evolution)
        - Movement bonus: distance traveled (discourages camping)
        """
        kill_fitness = agent.kills * 100
        
        max_possible_dist = math.hypot(self.width, self.height)
        proximity_bonus = (1 - min(agent.closest_enemy_distance / max_possible_dist, 1.0)) * 500
        
        movement_bonus = agent.total_distance * 0.05
        
        return kill_fitness + proximity_bonus + movement_bonus

    def _handle_events(self):
        """Listens for quit signals from Pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()