import neat
import pygame
import numpy as np
import random
import math
import csv
import os
from dataclasses import astuple

from src.models.prey import Prey
from src.models.predator import Predator
from src.models.food import Food
from src.core.sim_controller import SimulationController
from src.core.render_controller import RenderController
import src.configs.sim_config as sim_config
from src.configs.nn_config import NNOutputs

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

            for genome, net, agent in prey_nets:
                genome.fitness = self._calculate_prey_fitness(agent)
                
            for genome, net, agent in predator_nets:
                genome.fitness = self._calculate_predator_fitness(agent)

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
            agent = Prey(random.randint(center_x_min, center_x_max), 
                         random.randint(center_y_min, center_y_max))
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
                
            agent = Predator(px, py)
            predator_nets.append((genome, net, agent))
            self.sim.predators.append(agent)
        
        self.add_food()
            
        generation_frames = 0
        max_frames = sim_config.MAX_FRAMES
        is_running = True
        clock = pygame.time.Clock()
        max_distance = math.sqrt(self.width ** 2 + self.height ** 2)
        
        while is_running and generation_frames < max_frames:
            if self.render_enabled:
                self._handle_events()
            
            missing_food = sim_config.FOOD_COUNT - len(self.sim.foods)
            if missing_food:
                self.add_food(missing_food)
            
            # INTENTION (Move and Record State)
            for genome, net, agent in prey_nets:
                if agent.is_alive:
                    agent.foods_before_step = agent.foods 
                    
                    inputs = self.sim.get_agent_inputs(agent, self.sim.predators, self.sim.foods)
                    outputs = net.activate(list(astuple(inputs)))
                    agent.apply_action(NNOutputs(*outputs), self.width, self.height)

            for genome, net, agent in predator_nets:
                if agent.is_alive:
                    agent.hunt_score += (1.0 - inputs.enemy_distance)
                    agent.kills_before_step = agent.kills 
                    
                    inputs = self.sim.get_agent_inputs(agent, self.sim.preys, self.sim.foods)
                    outputs = net.activate(list(astuple(inputs)))
                    agent.apply_action(NNOutputs(*outputs), self.width, self.height)

            # RESOLUTION (Physics & Collisions)
            self.sim.update_physics()
            
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
            
            if render_this_gen:
                self.renderer.draw_frame(self.sim)
                clock.tick(self.fps)
                
            generation_frames += 1
            
            if len(self.sim.preys) == 0:
                is_running = False

        for genome, net, agent in prey_nets:
            genome.fitness = self._calculate_prey_fitness(agent)
            
        for genome, net, agent in predator_nets:
            genome.fitness = self._calculate_predator_fitness(agent)

        return prey_nets, predator_nets

    def _calculate_prey_fitness(self, agent) -> float:
        """Calculate fitness for a prey agent."""
        fitness = (agent.foods * 500) + agent.survival_frames - (agent.wall_frames * 5)
        return max(0.001, fitness)

    def _calculate_predator_fitness(self, agent) -> float:
        """Calculate fitness for a predator agent."""
        kill_score = agent.kills * 1000
        hunt_pressure = agent.hunt_score
        wall_penalty = agent.wall_frames * 5
        fitness = kill_score + hunt_pressure - wall_penalty
        return max(0.001, fitness)
    
    def add_food(self, n = sim_config.FOOD_COUNT):
        for _ in range(n):
            food = Food(random.randint(0, self.width),
                        random.randint(0, self.height),
                        self.sim.preys[0].radius / 2)
            self.sim.foods.append(food)

    def _handle_events(self):
        """Listens for quit signals from Pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

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

        stats = {
            "generation": generation,
            "avg_prey_survival": round(avg_prey_survival, 2),
            "max_prey_survival": max_prey_survival,
            "total_food": total_food,
            "avg_predator_kills": round(avg_predator_kills, 2),
            "max_predator_kills": max_predator_kills,
            "total_kills": total_kills
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