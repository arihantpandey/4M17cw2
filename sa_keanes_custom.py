import numpy as np
import math
from scipy.optimize import minimize_scalar

# Define the Keane's Bump Function for 8 dimensions
def keanes_bump_function(x):
    # Constraint check
    if np.any(x <= 0) or np.any(x >= 10):
        return np.inf  # Return infinity if constraints are violated
    prod_x = np.prod(x)
    if prod_x <= 0.75 or np.sum(x) >= 15 * len(x) / 2:
        return np.inf  # Return infinity if constraints are violated
    
    sum_cos_x4 = np.sum(np.cos(x) ** 4)
    prod_cos_x2 = np.prod(np.cos(x) ** 2)
    sum_xi_squared = np.sum(x ** 2)
    
    numerator = sum_cos_x4 - 2 * prod_cos_x2
    denominator = math.sqrt(sum_xi_squared)
    
    return -numerator / denominator  # Negative because we need to maximize

# Simulated annealing parameters
initial_temp = 10000
final_temp = 0.1
alpha = 0.99
max_iterations = 1000

# Simulated annealing algorithm
def simulated_annealing():
    # Initial random solution
    current_solution = np.random.uniform(0, 10, 8)
    current_value = keanes_bump_function(current_solution)
    current_temp = initial_temp
    
    best_solution = np.copy(current_solution)
    best_value = current_value

    for iteration in range(max_iterations):
        for i in range(8):  # Try perturbing each dimension
            next_solution = np.copy(current_solution)
            perturbation = np.random.uniform(-1, 1)
            next_solution[i] += perturbation  # Perturb the i-th dimension
            next_value = keanes_bump_function(next_solution)
            
            # Acceptance probability
            ap = np.exp((current_value - next_value) / current_temp)
            
            # Decide if we should accept the new solution
            if next_value < current_value or np.random.rand() < ap:
                current_solution, current_value = next_solution, next_value
                if current_value < best_value:
                    best_solution, best_value = current_solution, current_value
        
        # Cool down
        current_temp *= alpha
        
        # Stop if temperature is low
        if current_temp < final_temp:
            break
    
    return best_solution, -best_value

# Run simulated annealing
best_solution, best_value = simulated_annealing()
print("Best solution:", best_solution)
print("Best value:", best_value)
