import pygame
import random
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from itertools import product
from tqdm import tqdm
import os
from datetime import datetime
from joblib import Parallel, delayed

# GUI Snake Game for Visualization
class SnakeGameGUI:
    def __init__(self, width=500, height=500):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI - Best Model Performance")
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.snake_start = [100, 100]
        self.snake_position = [[100, 100], [90, 100], [80, 100]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.steps = 0
        self.max_steps = 2000
        return self.get_state()
    
    def get_state(self):
        head = self.snake_position[0]
        
        # Basic state (compatible with legacy models)
        apple_dist = np.linalg.norm(np.array(self.apple_position) - np.array(head))
        
        if len(self.snake_position) > 1:
            direction = np.array(head) - np.array(self.snake_position[1])
        else:
            direction = np.array([10, 0])
        
        dangers = self.get_dangers(head, direction)
        
        apple_dir = np.array(self.apple_position) - np.array(head)
        apple_angle = math.atan2(apple_dir[1], apple_dir[0])
        snake_angle = math.atan2(direction[1], direction[0])
        relative_angle = apple_angle - snake_angle
        relative_angle = relative_angle / math.pi
        
        # Legacy state (8 inputs)
        legacy_state = [
            apple_dist / 500.0,
            relative_angle,
            len(self.snake_position) / 100.0,
            dangers[0],
            dangers[1],
            dangers[2],
            head[0] / 500.0,
            head[1] / 500.0,
        ]
        
        # Additional state for new models (2 extra inputs)
        if hasattr(self, 'extended_state') and self.extended_state:
            tail_dir = np.array(self.snake_position[-1]) - np.array(head)
            tail_dist = np.linalg.norm(tail_dir) / 500.0
            return np.array(legacy_state + [
                tail_dist,
                int(self.is_tail_in_way(head, self.apple_position))
            ])
        
        return np.array(legacy_state)

    def is_tail_in_way(self, head, apple):
        # Check if tail segments are between head and apple
        head_to_apple = np.array(apple) - np.array(head)
        apple_dist = np.linalg.norm(head_to_apple)
        direction = head_to_apple / apple_dist
        
        # Check points along the path to apple
        check_dist = 20  # Check every 20 pixels
        num_checks = int(apple_dist / check_dist)
        
        for i in range(1, num_checks):
            check_point = head + direction * (i * check_dist)
            # Check if any tail segment is near this point
            for segment in self.snake_position[1:]:
                if np.linalg.norm(np.array(segment) - check_point) < 15:
                    return True
        return False
    
    def get_dangers(self, head, direction):
        directions = [
            direction,
            np.array([-direction[1], direction[0]]),
            np.array([direction[1], -direction[0]])
        ]
        
        dangers = []
        for d in directions:
            next_pos = [head[0] + d[0], head[1] + d[1]]
            danger = (self.collision_with_boundaries(next_pos) or 
                     self.collision_with_self(next_pos))
            dangers.append(1.0 if danger else 0.0)
        
        return dangers
    
    def step(self, action):
        if len(self.snake_position) > 1:
            current_direction = np.array(self.snake_position[0]) - np.array(self.snake_position[1])
        else:
            current_direction = np.array([10, 0])
        
        if action == 1:  # left
            new_direction = np.array([-current_direction[1], current_direction[0]])
        elif action == 2:  # right
            new_direction = np.array([current_direction[1], -current_direction[0]])
        else:  # straight
            new_direction = current_direction
        
        self.snake_start[0] += new_direction[0]
        self.snake_start[1] += new_direction[1]
        
        reward = 0
        done = False
        
        if (self.collision_with_boundaries(self.snake_start) or 
            self.collision_with_self(self.snake_start)):
            reward = -100
            done = True
        elif self.snake_start == self.apple_position:
            self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
            self.score += 1
            reward = 100
            self.snake_position.insert(0, list(self.snake_start))
        else:
            self.snake_position.insert(0, list(self.snake_start))
            self.snake_position.pop()
            
            old_dist = np.linalg.norm(np.array(self.apple_position) - np.array(self.snake_position[1]))
            new_dist = np.linalg.norm(np.array(self.apple_position) - np.array(self.snake_start))
            if new_dist < old_dist:
                reward = 1
            else:
                reward = -1
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        return self.get_state(), reward, done
    
    def collision_with_boundaries(self, pos):
        return pos[0] >= self.width or pos[0] < 0 or pos[1] >= self.height or pos[1] < 0
    
    def collision_with_self(self, pos):
        return pos in self.snake_position[1:]
    
    def render(self):
        self.display.fill((0, 0, 0))  # Black background
        
        # Draw snake
        for segment in self.snake_position:
            pygame.draw.rect(self.display, (0, 255, 0), 
                           pygame.Rect(segment[0], segment[1], 10, 10))
        
        # Draw apple
        pygame.draw.rect(self.display, (255, 0, 0), 
                        pygame.Rect(self.apple_position[0], self.apple_position[1], 10, 10))
        
        # Display score and info
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        steps_text = font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        length_text = font.render(f"Length: {len(self.snake_position)}", True, (255, 255, 255))
        
        self.display.blit(score_text, (10, 10))
        self.display.blit(steps_text, (10, 50))
        self.display.blit(length_text, (10, 90))
        
        pygame.display.update()
        self.clock.tick(10)  # Adjust speed as needed
class SnakeGameHeadless:
    def __init__(self, width=500, height=500):
        self.width = width
        self.height = height
        self.extended_state = False  # Add this flag
        self.prev_distance_to_apple = None
        self.position_history = []
        self.no_progress_counter = 0
        self.last_apple_distance = None
        self.reset()
    
    def reset(self):
        self.snake_start = [100, 100]
        self.snake_position = [[100, 100], [90, 100], [80, 100]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.steps = 0
        self.max_steps = 1000
        self.position_history = []
        self.no_progress_counter = 0
        self.last_apple_distance = None
        return self.get_state()
    
    def get_state(self):
        # Enhanced state representation
        head = self.snake_position[0]
        
        # Distance to apple
        apple_dist = np.linalg.norm(np.array(self.apple_position) - np.array(head))
        
        # Direction vectors
        if len(self.snake_position) > 1:
            direction = np.array(head) - np.array(self.snake_position[1])
        else:
            direction = np.array([10, 0])
        
        # Danger detection (collision in next step)
        dangers = self.get_dangers(head, direction)
        
        # Apple direction relative to snake
        apple_dir = np.array(self.apple_position) - np.array(head)
        apple_angle = math.atan2(apple_dir[1], apple_dir[0])
        snake_angle = math.atan2(direction[1], direction[0])
        relative_angle = apple_angle - snake_angle
        
        # Normalize angle to [-1, 1]
        relative_angle = relative_angle / math.pi
        
        state = [
            apple_dist / 500.0,  # Normalized distance to apple
            relative_angle,      # Relative angle to apple
            len(self.snake_position) / 100.0,  # Snake length
            dangers[0],          # Danger straight
            dangers[1],          # Danger left
            dangers[2],          # Danger right
            head[0] / 500.0,     # Head x position (normalized)
            head[1] / 500.0,     # Head y position (normalized)
        ]
        
        return np.array(state)
    
    def get_dangers(self, head, direction):
        # Check danger in three directions: straight, left, right
        directions = [
            direction,  # straight
            np.array([-direction[1], direction[0]]),  # left (90° counterclockwise)
            np.array([direction[1], -direction[0]])   # right (90° clockwise)
        ]
        
        dangers = []
        for d in directions:
            next_pos = [head[0] + d[0], head[1] + d[1]]
            danger = (self.collision_with_boundaries(next_pos) or 
                     self.collision_with_self(next_pos))
            dangers.append(1.0 if danger else 0.0)
        
        return dangers
    
    def get_distance_to_apple(self):
        head = self.snake_position[0]
        return np.linalg.norm(np.array(self.apple_position) - np.array(head))
    
    def distance_to_wall(self):
        head = self.snake_position[0]
        return min(
            head[0],  # distance to left wall
            self.width - head[0],  # distance to right wall
            head[1],  # distance to top wall
            self.height - head[1]  # distance to bottom wall
        )
    
    def calculate_fitness(self):
        score = self.score
        steps = self.steps
        
        # Base fitness calculation
        if score == 0:
            fitness = steps * 0.1  # Small reward for survival without apples
        else:
            fitness = (score ** 2) * 100 + steps * 0.5  # Quadratic reward for apples
        
        # Wall proximity penalty
        if self.distance_to_wall() < 30:  # 30 pixels from wall
            fitness -= 10
        
        # Apple approach reward/penalty
        curr_dist = self.get_distance_to_apple()
        if self.prev_distance_to_apple is not None:
            if curr_dist < self.prev_distance_to_apple:
                fitness += 5  # Reward for moving toward apple
            else:
                fitness -= 2  # Small penalty for moving away
        
        self.prev_distance_to_apple = curr_dist
        
        return fitness
    
    def is_snake_stuck(self):
        # Check last 50 positions for repeated patterns
        if len(self.position_history) > 50:
            last_positions = self.position_history[-50:]
            # If snake is moving in small area
            unique_positions = set(map(tuple, last_positions))
            if len(unique_positions) < 10:  # Snake revisiting same spots
                return True
            
        # Check if snake isn't making progress towards apple
        if self.last_apple_distance is not None:
            current_distance = np.linalg.norm(
                np.array(self.apple_position) - np.array(self.snake_position[0])
            )
            # If distance hasn't decreased in last 50 moves
            if current_distance >= self.last_apple_distance:
                self.no_progress_counter += 1
                if self.no_progress_counter > 50:
                    return True
            else:
                self.no_progress_counter = 0
            self.last_apple_distance = current_distance
            
        return False

    def step(self, action):
        # Store previous distance for comparison
        self.prev_distance_to_apple = self.get_distance_to_apple()
        
        # Convert neural network output to direction
        if len(self.snake_position) > 1:
            current_direction = np.array(self.snake_position[0]) - np.array(self.snake_position[1])
        else:
            current_direction = np.array([10, 0])
        
        # Action: 0=straight, 1=left, 2=right
        if action == 1:  # left
            new_direction = np.array([-current_direction[1], current_direction[0]])
        elif action == 2:  # right
            new_direction = np.array([current_direction[1], -current_direction[0]])
        else:  # straight
            new_direction = current_direction
        
        # Update snake position
        self.snake_start[0] += new_direction[0]
        self.snake_start[1] += new_direction[1]
        
        reward = 0
        done = False
        
        # Check collisions
        if (self.collision_with_boundaries(self.snake_start) or 
            self.collision_with_self(self.snake_start)):
            reward = -100
            done = True
        elif self.snake_start == self.apple_position:
            # Ate apple
            self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
            self.score += 1
            reward = 100
            self.snake_position.insert(0, list(self.snake_start))
        else:
            # Normal move
            self.snake_position.insert(0, list(self.snake_start))
            self.snake_position.pop()
            
            # Small reward for getting closer to apple
            old_dist = np.linalg.norm(np.array(self.apple_position) - np.array(self.snake_position[1]))
            new_dist = np.linalg.norm(np.array(self.apple_position) - np.array(self.snake_start))
            if new_dist < old_dist:
                reward = 1
            else:
                reward = -1
        
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        
        # Update fitness calculation
        reward = self.calculate_fitness()
        
        # Store position history
        self.position_history.append(self.snake_position[0].copy())
        if len(self.position_history) > 100:  # Keep last 100 positions
            self.position_history.pop(0)
            
        # Check for stuck condition
        if self.is_snake_stuck():
            done = True
            reward -= 50  # Penalty for getting stuck
            
        return self.get_state(), reward, done
    
    def collision_with_boundaries(self, pos):
        return pos[0] >= self.width or pos[0] < 0 or pos[1] >= self.height or pos[1] < 0
    
    def collision_with_self(self, pos):
        return pos in self.snake_position[1:]

# Neural Network with Xavier initialization
class NeuralNetwork:
    def __init__(self, weights=None, n_inputs=8, n_hidden1=12, n_hidden2=8, n_outputs=3):
        # Corrected initialization
        self.n_inputs = n_inputs
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.legacy_mode = False
        
        if weights is not None:
            expected_size_new = n_inputs * n_hidden1 + n_hidden1 * n_hidden2 + n_hidden2 * n_outputs
            legacy_size = 8 * 12 + 12 * 8 + 8 * 3
            
            if len(weights) == legacy_size:
                self.n_inputs = 8
                self.legacy_mode = True
                self.set_weights(weights)
            elif len(weights) == expected_size_new:
                self.legacy_mode = False
                self.set_weights(weights)
            else:
                raise ValueError(f"Incompatible weights size. Expected {expected_size_new} or {legacy_size}")
        else:
            self.randomize_weights()

    def randomize_weights(self):
        # Xavier initialization
        w1_size = self.n_inputs * self.n_hidden1
        w2_size = self.n_hidden1 * self.n_hidden2
        w3_size = self.n_hidden2 * self.n_outputs
        
        self.w1 = np.random.randn(self.n_inputs, self.n_hidden1) * np.sqrt(2.0 / self.n_inputs)
        self.w2 = np.random.randn(self.n_hidden1, self.n_hidden2) * np.sqrt(2.0 / self.n_hidden1)
        self.w3 = np.random.randn(self.n_hidden2, self.n_outputs) * np.sqrt(2.0 / self.n_hidden2)
    
    def set_weights(self, weights):
        w1_end = self.n_inputs * self.n_hidden1
        w2_end = w1_end + self.n_hidden1 * self.n_hidden2
        
        self.w1 = weights[:w1_end].reshape(self.n_inputs, self.n_hidden1)
        self.w2 = weights[w1_end:w2_end].reshape(self.n_hidden1, self.n_hidden2)
        self.w3 = weights[w2_end:].reshape(self.n_hidden2, self.n_outputs)
    
    def get_weights(self):
        return np.concatenate([self.w1.flatten(), self.w2.flatten(), self.w3.flatten()])
    
    def predict(self, x):
        # Handle legacy input format
        if self.legacy_mode and len(x) > 8:
            # Use only the first 8 inputs for legacy models
            x = x[:8]
        elif not self.legacy_mode and len(x) < self.n_inputs:
            # Pad with zeros for new models if needed
            x = np.pad(x, (0, self.n_inputs - len(x)))
            
        h1 = np.tanh(np.dot(x, self.w1))
        h2 = np.tanh(np.dot(h1, self.w2))
        output = np.dot(h2, self.w3)
        return np.argmax(output)

# Genetic Algorithm with comprehensive parameter testing
class GeneticAlgorithm:
    def __init__(self, pop_size=50, crossover_rate=0.8, mutation_rate=0.1, 
                 crossover_type='one_point', num_weights=None):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type
        self.num_weights = num_weights or (8*12 + 12*8 + 8*3)
        
        # Initialize population
        self.population = np.random.uniform(-1, 1, (pop_size, self.num_weights))
    
    def evaluate_fitness(self, individual):
        nn = NeuralNetwork(weights=individual)
        game = SnakeGameHeadless()
        
        total_fitness = 0
        num_games = 3
        
        for _ in range(num_games):
            state = game.reset()
            done = False
            game_fitness = 0
            prev_action = None
            direction_changes = 0
            
            while not done and game.steps < 1000:
                action = nn.predict(state)
                
                # Penalize frequent direction changes
                if prev_action is not None and action != prev_action:
                    direction_changes += 1
                prev_action = action
                
                state, reward, done = game.step(action)
                
                # Add movement smoothness reward
                if not done:
                    game_fitness += reward - (direction_changes * 0.5)
                    
                    # Bonus for maintaining straight movement toward apple
                    if action == 0 and reward > 0:  # Going straight and getting closer
                        game_fitness += 2
        
        # Additional rewards/penalties
        if game.score > 0:
            game_fitness += (game.score ** 2) * 50
            # Bonus for efficient paths (less direction changes)
            efficiency_bonus = max(0, 100 - direction_changes) * game.score
            game_fitness += efficiency_bonus
        
        if game.steps < 100:
            game_fitness *= 0.5
        
        total_fitness += game_fitness
        return (total_fitness / num_games)
    
    def selection(self, fitness_scores):
        # Tournament selection
        selected = []
        for _ in range(self.pop_size):
            tournament_size = 3
            tournament_indices = np.random.choice(len(fitness_scores), tournament_size)
            winner_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
            selected.append(self.population[winner_idx].copy())
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.crossover_type == 'one_point':
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        else:  # two_point
            point1 = random.randint(1, len(parent1) - 2)
            point2 = random.randint(point1 + 1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
            child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])
        
        return child1, child2
    
    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
        return individual
    
    def evolve(self, generations=100, verbose=False):
        best_fitness_history = []
        
        for gen in range(generations):
            # Evaluate fitness
            fitness_scores = np.array([self.evaluate_fitness(ind) for ind in self.population])
            best_fitness_history.append(np.max(fitness_scores))
            
            if verbose and gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {np.max(fitness_scores):.2f}")
            
            # Selection
            selected = self.selection(fitness_scores)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, self.pop_size, 2):
                parent1, parent2 = selected[i], selected[(i + 1) % self.pop_size]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            self.population = np.array(new_population[:self.pop_size])
        
        # Return best individual and fitness history
        final_fitness = np.array([self.evaluate_fitness(ind) for ind in self.population])
        best_idx = np.argmax(final_fitness)
        return self.population[best_idx], best_fitness_history

# Particle Swarm Optimization with full parameter options
class PSO:
    def __init__(self, pop_size=50, w=0.7, c1=1.5, c2=1.5, num_weights=None):
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Determine number of weights based on legacy/new mode
        self.legacy_mode = True  # Default to legacy mode for compatibility
        self.num_weights = num_weights or (8 * 12 + 12 * 8 + 8 * 3)  # Legacy size
        
        # Initialize particles
        self.particles = np.random.uniform(-0.5, 0.5, (pop_size, self.num_weights))
        self.velocities = np.random.uniform(-0.1, 0.1, (pop_size, self.num_weights))
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(pop_size, -np.inf)
        self.global_best = None
        self.global_best_fitness = -np.inf
    
    def evaluate_fitness(self, individual):
        nn = NeuralNetwork(weights=individual)
        game = SnakeGameHeadless()
        
        total_fitness = 0
        num_games = 3
        
        for _ in range(num_games):
            state = game.reset()
            done = False
            game_fitness = 0
            
            while not done and game.steps < 1000:
                action = nn.predict(state)
                state, reward, done = game.step(action)
                game_fitness += reward
            
            # Additional rewards/penalties
            if game.score > 0:
                game_fitness += (game.score ** 2) * 50
            if game.steps < 100:
                game_fitness *= 0.5
            
            total_fitness += game_fitness
        
        return total_fitness / num_games
    
    def optimize(self, generations=100, verbose=False):
        best_fitness_history = []
        
        for gen in range(generations):
            # Evaluate all particles
            for i in range(self.pop_size):
                fitness = self.evaluate_fitness(self.particles[i])
                
                # Update personal best
                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].copy()
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.particles[i].copy()
            
            best_fitness_history.append(self.global_best_fitness)
            
            if verbose and gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {self.global_best_fitness:.2f}")
            
            # Update velocities and positions
            for i in range(self.pop_size):
                r1, r2 = np.random.random(), np.random.random()
                
                self.velocities[i] = (self.w * self.velocities[i] + 
                                    self.c1 * r1 * (self.personal_best[i] - self.particles[i]) +
                                    self.c2 * r2 * (self.global_best - self.particles[i]))
                
                self.particles[i] += self.velocities[i]
                
                # Clamp particles to bounds
                self.particles[i] = np.clip(self.particles[i], -2, 2)
        
        return self.global_best, best_fitness_history

# Training System with comprehensive parameter testing and parallel processing
class SnakeAITrainer:
    def __init__(self):
        self.results = []
        
        # Reduced parameter ranges
        self.pop_sizes = list(range(20, 201, 20))  # 10 values: 20,40,60,...,200
        self.crossover_rates = [0.2, 0.4, 0.6, 0.8, 1.0]  # 5 values
        self.mutation_rates = [0.01, 0.03, 0.05, 0.07, 0.1]  # 5 values
        self.crossover_types = ['one_point', 'two_point']  # 2 values
        
        # Reduced PSO parameters
        self.pso_w_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # 5 values
        self.pso_c1_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 values
        self.pso_c2_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 values
    
    def train_ga_combination(self, comb, generations):
        pop_size, crossover_rate, mutation_rate, crossover_type = comb
        try:
            ga = GeneticAlgorithm(
                pop_size=pop_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                crossover_type=crossover_type
            )
            best_individual, fitness_history = ga.evolve(generations=generations)
            max_fitness = max(fitness_history)
            avg_fitness = np.mean(fitness_history[-10:])  # Average of last 10 generations
            
            return {
                'algorithm': 'GA',
                'pop_size': pop_size,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate,
                'crossover_type': crossover_type,
                'max_fitness': max_fitness,
                'avg_fitness': avg_fitness,
                'best_weights': best_individual,
                'fitness_history': fitness_history
            }
        except Exception as e:
            print(f"Error in GA combination {comb}: {e}")
            return None
    
    def train_genetic_algorithm(self, generations=50):
        # Define parameter combinations
        param_combinations = list(product(
            [100, 150, 200],  # population size
            [0.2, 0.4, 0.6, 0.8],  # crossover rate
            [0.04, 0.08, 0.12],  # mutation rate
            ['one_point', 'two_point']  # crossover type
        ))

        print(f"Testing {len(param_combinations)} GA parameter combinations...")
        
        # Run parallel optimization for each combination
        results = Parallel(n_jobs=-1)(
            delayed(self.train_ga_combination)(comb, generations) 
            for comb in tqdm(param_combinations, desc="GA Combinations")
        )

        self.save_results(results, "GA")
        self.plot_results(results, "GA")
        return results

    def train_pso_combination(self, comb, generations):
        pop_size, w, c1, c2 = comb
        try:
            pso = PSO(pop_size=pop_size, w=w, c1=c1, c2=c2)
            best_individual, fitness_history = pso.optimize(generations=generations)
            max_fitness = max(fitness_history)
            avg_fitness = np.mean(fitness_history[-10:])
            
            return {
                'algorithm': 'PSO',
                'pop_size': pop_size,
                'w': w,
                'c1': c1,
                'c2': c2,
                'max_fitness': max_fitness,
                'avg_fitness': avg_fitness,
                'best_weights': best_individual,
                'fitness_history': fitness_history
            }
        except Exception as e:
            print(f"Error in PSO combination {comb}: {e}")
            return None
    
    def train_pso(self, generations=50):
        # Define parameter combinations
        param_combinations = list(product(
            [100, 150, 200],  # population size
            [0.2, 0.4, 0.6],  # w (inertia weight)
            [1.0, 2.0, 3.0],  # c1 (cognitive parameter)
            [1.0, 2.0, 3.0]   # c2 (social parameter)
        ))

        print(f"Testing {len(param_combinations)} PSO parameter combinations...")
        
        # Run parallel optimization for each combination
        results = Parallel(n_jobs=-1)(
            delayed(self.train_pso_combination)(comb, generations) 
            for comb in tqdm(param_combinations, desc="PSO Combinations")
        )

        self.save_results(results, "PSO")
        self.plot_results(results, "PSO")
        return results

    def save_results(self, results, algorithm_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Add mode information to results
        for result in results:
            result['extended_state'] = getattr(SnakeGameHeadless, 'extended_state', False)
        
        # Save detailed results
        filename = f"snake_ai_{algorithm_type.lower()}_{'extended' if getattr(SnakeGameHeadless, 'extended_state', False) else 'legacy'}_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    
        # Save best model
        if results:
            best_result = max(results, key=lambda x: x['max_fitness'])
            best_filename = f"best_snake_ai_{algorithm_type.lower()}_{timestamp}.pkl"
            with open(best_filename, 'wb') as f:
                pickle.dump(best_result, f)
        
            print(f"Best {algorithm_type} model saved: {best_filename}")
            print(f"Best fitness: {best_result['max_fitness']:.2f}")
    
        # Create results table
        df = pd.DataFrame(results)
        csv_filename = f"snake_ai_{algorithm_type.lower()}_table_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
    
        return results
    
    def plot_results(self, results, algorithm_type):
        print(f"Generating plots for {algorithm_type} results...")
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Set non-interactive backend
            plt.switch_backend('Agg')
            
            # Create plots with progress indicators
            print("Creating parameter analysis plot...")
            fig1, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            df = pd.DataFrame(results)
            
            if algorithm_type == 'GA':
                # Population size vs fitness
                pop_fitness = df.groupby('pop_size')['max_fitness'].mean()
                axes[0,0].plot(pop_fitness.index, pop_fitness.values, 'bo-')
                axes[0,0].set_xlabel('Population Size')
                axes[0,0].set_ylabel('Average Max Fitness')
                axes[0,0].set_title('Population Size vs Fitness')
                
                # Crossover rate vs fitness
                cross_fitness = df.groupby('crossover_rate')['max_fitness'].mean()
                axes[0,1].plot(cross_fitness.index, cross_fitness.values, 'ro-')
                axes[0,1].set_xlabel('Crossover Rate')
                axes[0,1].set_ylabel('Average Max Fitness')
                axes[0,1].set_title('Crossover Rate vs Fitness')
                
                # Mutation rate vs fitness
                mut_fitness = df.groupby('mutation_rate')['max_fitness'].mean()
                axes[1,0].plot(mut_fitness.index, mut_fitness.values, 'go-')
                axes[1,0].set_xlabel('Mutation Rate')
                axes[1,0].set_ylabel('Average Max Fitness')
                axes[1,0].set_title('Mutation Rate vs Fitness')
                
                # Crossover type comparison
                type_fitness = df.groupby('crossover_type')['max_fitness'].mean()
                axes[1,1].bar(type_fitness.index, type_fitness.values)
                axes[1,1].set_xlabel('Crossover Type')
                axes[1,1].set_ylabel('Average Max Fitness')
                axes[1,1].set_title('Crossover Type vs Fitness')
            
            else:  # PSO
                # Population size vs fitness
                pop_fitness = df.groupby('pop_size')['max_fitness'].mean()
                axes[0,0].plot(pop_fitness.index, pop_fitness.values, 'bo-')
                axes[0,0].set_xlabel('Population Size')
                axes[0,0].set_ylabel('Average Max Fitness')
                axes[0,0].set_title('Population Size vs Fitness')
                
                # W parameter vs fitness
                w_fitness = df.groupby('w')['max_fitness'].mean()
                axes[0,1].plot(w_fitness.index, w_fitness.values, 'ro-')
                axes[0,1].set_xlabel('Inertia Weight (w)')
                axes[0,1].set_ylabel('Average Max Fitness')
                axes[0,1].set_title('Inertia Weight vs Fitness')
                
                # C1 parameter vs fitness
                c1_fitness = df.groupby('c1')['max_fitness'].mean()
                axes[1,0].plot(c1_fitness.index, c1_fitness.values, 'go-')
                axes[1,0].set_xlabel('Cognitive Parameter (c1)')
                axes[1,0].set_ylabel('Average Max Fitness')
                axes[1,0].set_title('C1 vs Fitness')
                
                # C2 parameter vs fitness
                c2_fitness = df.groupby('c2')['max_fitness'].mean()
                axes[1,1].plot(c2_fitness.index, c2_fitness.values, 'mo-')
                axes[1,1].set_xlabel('Social Parameter (c2)')
                axes[1,1].set_ylabel('Average Max Fitness')
                axes[1,1].set_title('C2 vs Fitness')
            
            plt.tight_layout()
            # Save instead of show
            plot_filename = f'snake_ai_{algorithm_type.lower()}_analysis_{timestamp}.png'
            fig1.savefig(plot_filename)
            plt.close(fig1)
            print(f"Analysis plot saved as: {plot_filename}")
            
            # Best evolution plot
            print("Creating evolution plot...")
            fig2 = plt.figure(figsize=(10, 6))
            best_result = max(results, key=lambda x: x['max_fitness'])
            plt.plot(best_result['fitness_history'], 'b-', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.title(f'Best {algorithm_type} Model - Fitness Evolution')
            plt.grid(True, alpha=0.3)
            
            evolution_filename = f'best_{algorithm_type.lower()}_evolution_{timestamp}.png'
            fig2.savefig(evolution_filename)
            plt.close(fig2)
            print(f"Evolution plot saved as: {evolution_filename}")
        
        except Exception as e:
            print(f"Error during plotting: {str(e)}")
        finally:
            plt.close('all')  # Ensure all plots are closed

def load_and_test_model(model_path):
    """Load a trained model and test it with GUI visualization"""
    print(f"Loading model from: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print("Model loaded successfully!")
        print(f"Algorithm: {model_data['algorithm']}")
        print(f"Max Fitness: {model_data['max_fitness']:.2f}")
        
        if model_data['algorithm'] == 'GA':
            print(f"Population Size: {model_data['pop_size']}")
            print(f"Crossover Rate: {model_data['crossover_rate']}")
            print(f"Mutation Rate: {model_data['mutation_rate']}")
            print(f"Crossover Type: {model_data['crossover_type']}")
        else:  # PSO
            print(f"Population Size: {model_data['pop_size']}")
            print(f"Inertia Weight (w): {model_data['w']}")
            print(f"Cognitive Parameter (c1): {model_data['c1']}")
            print(f"Social Parameter (c2): {model_data['c2']}")
        
        # Create neural network with loaded weights
        nn = NeuralNetwork(weights=model_data['best_weights'])
        
        # Start GUI game
        game = SnakeGameGUI()
        
        print("\nStarting GUI demonstration...")
        print("Close the game window to exit.")
        print("Controls:")
        print("  - SPACE: Start new game")
        print("  - ESC: Exit")
        print("The AI will play automatically!")
        
        running = True
        game_count = 0
        total_scores = []
        
        while running:
            state = game.reset()
            done = False
            game_count += 1
            
            print(f"\nGame {game_count} started")
            
            while not done and running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            # Space bar to start new game
                            done = True
                        elif event.key == pygame.K_ESCAPE:
                            # Escape to exit
                            running = False
                
                if not running:
                    break
                
                # AI makes decision
                action = nn.predict(state)
                state, reward, done = game.step(action)
                
                # Render the game
                game.render()
                
                # Optional: print AI decision info
                if game.steps % 50 == 0:  # Print every 50 steps
                    action_names = ['Straight', 'Left', 'Right']
                    print(f"Step {game.steps}: Action = {action_names[action]}, Score = {game.score}")
            
            if running:
                final_score = game.score
                total_scores.append(final_score)
                print(f"Game {game_count} ended - Final Score: {final_score}")
                print(f"Average Score so far: {np.mean(total_scores):.2f}")
                
                # Auto-restart after a brief pause
                time.sleep(1)
        
        pygame.quit()
        
        if total_scores:
            print(f"\nFinal Statistics:")
            print(f"Games Played: {len(total_scores)}")
            print(f"Average Score: {np.mean(total_scores):.2f}")
            print(f"Best Score: {max(total_scores)}")
            print(f"Worst Score: {min(total_scores)}")
        
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Make sure you have trained and saved a model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

def list_saved_models():
    """List all saved model files in current directory"""
    model_files = [f for f in os.listdir('.') if f.startswith('best_snake_ai_') and f.endswith('.pkl')]
    
    if not model_files:
        print("No saved models found.")
        print("Train a model first using the main training function.")
        return []
    
    print("Available saved models:")
    for i, filename in enumerate(model_files, 1):
        # Extract info from filename
        parts = filename.replace('best_snake_ai_', '').replace('.pkl', '').split('_')
        algorithm = parts[0].upper()
        timestamp = parts[1] if len(parts) > 1 else "unknown"
        print(f"{i}. {filename} ({algorithm} - {timestamp})")
    
    return model_files

def main_menu():
    """Interactive menu system for the Snake AI Trainer"""
    while True:
        print("\n===== Snake AI Trainer =====")
        print("1. Train new models")
        print("2. Test saved model with GUI")
        print("3. List saved models")
        print("4. Toggle Extended State Mode")  # Add this option
        print("5. Exit")
        
        choice = input("Enter your choice: ").strip()
        
        if choice == '1':
            trainer = SnakeAITrainer()
            algorithm_choice = input("Choose algorithm (GA/PSO/BOTH): ").upper()
            generations = int(input("Enter number of generations (default 50): ") or "50")
            
            if algorithm_choice == "GA":
                results = trainer.train_genetic_algorithm(generations=generations)
                trainer.save_results(results, "GA")
                trainer.plot_results(results, "GA")
            elif algorithm_choice == "PSO":
                results = trainer.train_pso(generations=generations)
                trainer.save_results(results, "PSO")
                trainer.plot_results(results, "PSO")
            else:
                print("Training both algorithms...")
                
                # Train GA
                ga_results = trainer.train_genetic_algorithm(generations=generations)
                trainer.save_results(ga_results, "GA")
                trainer.plot_results(ga_results, "GA")
                
                # Train PSO
                pso_results = trainer.train_pso(generations=generations)
                trainer.save_results(pso_results, "PSO")
                trainer.plot_results(pso_results, "PSO")
                
                # Compare best from both
                if ga_results and pso_results:
                    best_ga = max(ga_results, key=lambda x: x['max_fitness'])
                    best_pso = max(pso_results, key=lambda x: x['max_fitness'])
                    
                    print("\n" + "="*50)
                    print("COMPARISON RESULTS")
                    print("="*50)
                    print(f"Best GA Fitness: {best_ga['max_fitness']:.2f}")
                    print(f"Best PSO Fitness: {best_pso['max_fitness']:.2f}")
                    
                    if best_ga['max_fitness'] > best_pso['max_fitness']:
                        print("Genetic Algorithm performed better!")
                    else:
                        print("PSO performed better!")
        
        elif choice == '2':
            model_files = list_saved_models()
            if model_files:
                try:
                    model_choice = int(input("Enter the number of the model to test: "))
                    if 1 <= model_choice <= len(model_files):
                        model_path = model_files[model_choice-1]
                        load_and_test_model(model_path)
                    else:
                        print("Invalid model number.")
                except ValueError:
                    print("Please enter a valid number.")
        
        elif choice == '3':
            list_saved_models()
        
        elif choice == '4':
            current_mode = "extended" if hasattr(SnakeGameHeadless, "extended_state") and SnakeGameHeadless.extended_state else "legacy"
            print(f"Current mode: {current_mode}")
            toggle = input("Toggle to extended state mode? (y/n): ").lower() == 'y'
            SnakeGameHeadless.extended_state = toggle
            print(f"Mode set to: {'extended' if toggle else 'legacy'}")
        
        elif choice == '5':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1-4.")

if __name__ == "__main__":
    main_menu()