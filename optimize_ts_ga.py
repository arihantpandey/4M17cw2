import random
import numpy as np
from ts_callable import tabu_search
# Define the Keane's Bump Function and Tabu Search here (as previously defined)

def initialize_population(size, param_bounds, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    population = []
    for _ in range(size):
        individual = {
            'max_iterations': random.randint(*param_bounds['max_iterations']),
            'tabu_list_size': random.randint(*param_bounds['tabu_list_size']),
            'neighborhood_size': random.randint(*param_bounds['neighborhood_size']),
            'intensification_factor': random.uniform(*param_bounds['intensification_factor']),
            'diversification_threshold': random.uniform(*param_bounds['diversification_threshold']),
            'initial_step_size': random.uniform(*param_bounds['initial_step_size']),
            'step_size_reduction_factor': random.uniform(*param_bounds['step_size_reduction_factor'])
        }
        population.append(individual)
    return population

def fitness(individual):
    # Run Tabu Search with the individual's hyperparameters and return the performance
    
    _, best_value = tabu_search(**individual)
    print(f"solution: {best_value}")
    return best_value

def select_parents(population, fitnesses, num_parents):
    # Ensure an even number of parents for crossover
    if num_parents % 2 != 0:
        num_parents -= 1

    fitness_sum = sum(fitnesses)
    selection_probs = [f / fitness_sum for f in fitnesses]
    parents = np.random.choice(population, size=num_parents, replace=False, p=selection_probs)
    return parents

def crossover(parent1, parent2):
    # Simple one-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = {**parent1}
    child2 = {**parent2}
    for i, key in enumerate(parent1.keys()):
        if i >= crossover_point:
            child1[key], child2[key] = child2[key], child1[key]
    return child1, child2

def mutate(individual, param_bounds, mutation_rate):
    for key in individual:
        if random.random() < mutation_rate:
            if isinstance(individual[key], int):
                individual[key] = random.randint(*param_bounds[key])
            else:
                individual[key] = random.uniform(*param_bounds[key])

def evolutionary_algorithm():
    param_bounds = {
        'max_iterations': (100, 5000),
        'tabu_list_size': (50, 200),
        'neighborhood_size': (20, 100),
        'intensification_factor': (1.0, 5.0),
        'diversification_threshold': (0.1, 1.0),
        'initial_step_size': (0.1, 5.0),
        'step_size_reduction_factor': (0.8, 0.99)
    }

    population_size = 16
    num_generations = 12
    num_parents = population_size // 2
    mutation_rate = 0.1

    population = initialize_population(population_size, param_bounds, seed=42)
    best_individual = None
    best_fitness = float('-inf')
    archive = []  # Initialize an archive for best solutions
    archive_size = 5  # Max number of individuals to keep in the archive

    for generation in range(num_generations):
        fitnesses = [fitness(individual) for individual in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        generation_best_fitness = max(fitnesses)

        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = population[fitnesses.index(best_fitness)]
            archive.append(best_individual)  # Add the best individual to the archive

        # Keep only the top N individuals in the archive
        archive = sorted(archive, key=lambda ind: fitness(ind), reverse=True)[:archive_size]

        print(f"Generation {generation + 1}/{num_generations} - Best Fitness: {best_fitness}, Avg Fitness: {avg_fitness}, Archive Size: {len(archive)}")

        parents = select_parents(population, fitnesses, num_parents)
        next_generation = []

        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            mutate(child1, param_bounds, mutation_rate)
            mutate(child2, param_bounds, mutation_rate)
            next_generation.extend([child1, child2])

        population = next_generation

    return best_individual, archive

best_hyperparameters, archive = evolutionary_algorithm()
print("Best hyperparameters:", best_hyperparameters)

for i, archived_individual in enumerate(archive, start=1):
    print(f"Archived Individual {i}: {archived_individual}, Fitness: {fitness(archived_individual)}")

_, best_value = tabu_search(**best_hyperparameters)
print("Tabu Search Result with Best Hyperparameters:", best_value)