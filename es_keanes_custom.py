import numpy as np

# Keanes bump
def keanes_bump(x):
    if np.any(x < 0) or np.any(x > 10) or np.prod(x) <= 0.75 or np.sum(x) >= (15 * len(x)) / 2:
        return 0  
    term1 = np.sum(np.cos(x)**4)
    term2 = 2 * np.prod(np.cos(x)**2)
    indices = np.arange(1, len(x) + 1)
    term3 = np.sqrt(np.sum(indices * (x**2)))
    return -(np.abs((term1 - term2) / term3))  #maximize

# bounds for the 8 parameters
bounds = [(0, 10)] * 8

# parameters for the evolutionary algorithm
population_size = 50
generations = 100
mutation_rate = 0.1
recombination_rate = 0.7
dimension = 8

# Initialize the population with random values within the bounds
population = np.random.rand(population_size, dimension)
for i in range(dimension):
    population[:, i] = population[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

# Evolutionary algorithm
for generation in range(generations):
    # Evaluate the fitness of the population
    fitness = np.array([keanes_bump(ind) for ind in population])

    # Select parents - tournament selection
    parents = []
    for _ in range(population_size):
        # Randomly select two individuals and perform a tournament selection
        i, j = np.random.randint(0, population_size, 2)
        parents.append(population[i] if fitness[i] > fitness[j] else population[j])
    parents = np.array(parents)

    # Create offspring - discrete recombination and mutation
    offspring = np.empty((population_size, dimension))
    for i in range(population_size):
        # Discrete recombination
        parent1, parent2 = parents[np.random.choice(population_size, 2, replace=False)]
        mask = np.random.rand(dimension) < recombination_rate
        offspring[i] = np.where(mask, parent1, parent2)
        # Mutation
        mutation_mask = np.random.rand(dimension) < mutation_rate
        offspring[i] += mutation_mask * np.random.randn(dimension)

        # Ensure offspring are within bounds
        offspring[i] = np.clip(offspring[i], [b[0] for b in bounds], [b[1] for b in bounds])

    # Replace the old population with offspring
    population = offspring

# Find the best solution
best_fitness = np.max([keanes_bump(ind) for ind in population])
best_individual = population[np.argmax([keanes_bump(ind) for ind in population])]

print(best_individual)
print(best_fitness)
