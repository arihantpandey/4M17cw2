import numpy as np

# Define the Keane's Bump Function for 8 dimensions
def keanes_bump_function(x):
    # Constraint check
    if np.any(x <= 0) or np.any(x >= 10) or np.prod(x) <= 0.75 or np.sum(x) >= 15 * len(x) / 2:
        return np.inf  # Return infinity if constraints are violated
    
    sum_cos_x4 = np.sum(np.cos(x) ** 4)
    prod_cos_x2 = np.prod(np.cos(x) ** 2)
    sum_xi_squared = np.sum(x ** 2)
    
    numerator = sum_cos_x4 - 2 * prod_cos_x2
    denominator = np.sqrt(sum_xi_squared)
    
    return -numerator / denominator  # Negative because we need to maximize

# Initialize parameters
num_variables = 8
max_iterations = 1000
tabu_list_size = 100
neighborhood_size = 50
intensification_factor = 2  # Factor by which to intensify search around good solutions
diversification_threshold = 0.6  # Threshold to trigger diversification
step_size = 1.0  # Initial step size

# Initialize solution
best_solution = np.random.uniform(0, 10, num_variables)
best_value = keanes_bump_function(best_solution)

# Initialize memories
short_term_tabu_list = []
frequency_matrix = np.zeros((num_variables, 10))  # Long-term memory: frequency of visits
best_solutions_history = []  # Medium-term memory

# Function to update frequency matrix for long-term memory
def update_frequency_matrix(solution):
    # Ensure values are in the range 0 to 9
    discretized_solution = np.floor(solution * 0.9999).astype(int)
    for i, value in enumerate(discretized_solution):
        frequency_matrix[i][value] += 1


# Function to calculate diversification measure
def diversification_measure():
    return np.sum(frequency_matrix > (diversification_threshold * np.max(frequency_matrix)))

# Tabu Search algorithm
for iteration in range(max_iterations):
    neighborhood = [best_solution + np.random.uniform(-step_size, step_size, num_variables) for _ in range(neighborhood_size)]
    neighborhood = [np.clip(neighbor, 0, 10) for neighbor in neighborhood]  # Ensure solutions stay within bounds
    
    # Evaluate all solutions in the neighborhood
    neighborhood_values = [keanes_bump_function(neighbor) for neighbor in neighborhood]
    
    # Sort the neighborhood by the objective function value
    sorted_neighborhood = sorted(zip(neighborhood, neighborhood_values), key=lambda x: x[1])
    
        # Find the best candidate that is not in the short-term Tabu List
    next_solution = None
    for candidate, value in sorted_neighborhood:
        if list(candidate) not in short_term_tabu_list:
            next_solution = candidate
            next_value = value
            break
    
    # If no new solution is found (all are tabu), continue to the next iteration
    if next_solution is None:
        continue


    # Update the best solution found so far if the new candidate is better
    if next_value < best_value:
        best_solution, best_value = next_solution, next_value
        best_solutions_history.append((best_solution, best_value))  # Update medium-term memory

    # Update the short-term Tabu List
    short_term_tabu_list.append(list(next_solution))
    if len(short_term_tabu_list) > tabu_list_size:
        short_term_tabu_list.pop(0)

    # Update frequency matrix for long-term memory
    update_frequency_matrix(next_solution)

    # If the diversification measure is high, perform diversification
    if diversification_measure() > diversification_threshold:
        step_size *= intensification_factor  # Increase step size
        frequency_matrix *= 0.5  # Decay frequency matrix to forget older visits
    
    # If the current iteration is a multiple of some number, perform intensification
    if iteration % 100 == 0 and best_solutions_history:
        best_solution, best_value = min(best_solutions_history, key=lambda x: x[1])
        step_size /= intensification_factor  # Decrease step size for intensification

    # Gradually reduce step size for a finer search
    step_size *= 0.99

print("Best solution:", best_solution)
print("Best value:", -best_value)
