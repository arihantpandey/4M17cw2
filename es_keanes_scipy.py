from scipy.optimize import differential_evolution
import numpy as np

# Define the Keane's Bump function
def keanes_bump(x):
    if np.any(x <= 0) or np.any(x >= 10) or np.prod(x) <= 0.75 or np.sum(x) >= (15 * len(x)) / 2:
        return 0  
    
    term1 = np.sum(np.cos(x)**4)
    term2 = 2 * np.prod(np.cos(x)**2)
    indices = np.arange(1, len(x) + 1)
    term3 = np.sqrt(np.sum(indices * (x**2)))
    result = np.abs((term1 - term2) / term3)
    
    return -result

# Define bounds for each of the 8 parameters
bounds = [(0, 10)] * 8

# Define the recombination strategy parameters
strategy_params = {
    'strategy': 'best1bin',  # Strategy for producing new trial solutions
    'recombination': 0.7,    # Recombination constant
    'mutation': (0.5, 1.0),  # Mutation factor range
    'maxiter': 10000,         # Maximum number of generations over which entire population is evolved
    'popsize': 15,           # Multiplier for setting the total population size
    'tol': 0.01,             # Tolerance for convergence
    'updating': 'deferred'   # Updating strategy
}

# Run the differential evolution algorithm
result = differential_evolution(keanes_bump, bounds, **strategy_params)

# Print the result
print(result.x, -result.fun)
