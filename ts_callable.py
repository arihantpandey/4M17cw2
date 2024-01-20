import numpy as np


def keanes_bump_function(x):
    if (
        np.any(x <= 0)
        or np.any(x >= 10)
        or np.prod(x) <= 0.75
        or np.sum(x) >= 15 * len(x) / 2
    ):
        return np.inf  # Return infinity if constraints are violated
    sum_cos_x4 = np.sum(np.cos(x) ** 4)
    prod_cos_x2 = np.prod(np.cos(x) ** 2)
    sum_xi_squared = np.sum(x**2)
    numerator = sum_cos_x4 - 2 * prod_cos_x2
    denominator = np.sqrt(sum_xi_squared)
    return -numerator / denominator  # Negative because we need to maximize


def tabu_search(
    tabu_list_size,
    neighborhood_size,
    intensification_factor,
    diversification_threshold,
    initial_step_size,
    step_size_reduction_factor,
    num_variables=8,
    max_evaluations=10000,
):
    best_solution = np.random.uniform(0, 10, num_variables)
    best_value = keanes_bump_function(best_solution)
    evaluations = 1  # Start with the initial evaluation
    short_term_tabu_list = []
    frequency_matrix = np.zeros(
        (num_variables, 10)
    )  # Long-term memory: frequency of visits

    best_solutions_history = []  # Medium-term memory
    step_size = initial_step_size

    def update_frequency_matrix(solution):
        discretized_solution = np.floor(solution * 0.9999).astype(int)
        for i, value in enumerate(discretized_solution):
            frequency_matrix[i][value] += 1

    def diversification_measure():
        return np.sum(
            frequency_matrix > (diversification_threshold * np.max(frequency_matrix))
        )

    while evaluations < max_evaluations:
        neighborhood = [
            best_solution + np.random.uniform(-step_size, step_size, num_variables)
            for _ in range(neighborhood_size)
        ]
        neighborhood = [np.clip(neighbor, 0, 10) for neighbor in neighborhood]
        neighborhood_values = [
            keanes_bump_function(neighbor) for neighbor in neighborhood
        ]
        evaluations += (
            neighborhood_size  # Increment evaluations by the number of neighbors
        )

        sorted_neighborhood = sorted(
            zip(neighborhood, neighborhood_values), key=lambda x: x[1]
        )
        for candidate, value in sorted_neighborhood:
            if list(candidate) not in short_term_tabu_list:
                if value < best_value:
                    best_solution, best_value = candidate, value
                    best_solutions_history.append((best_solution, best_value))
                short_term_tabu_list.append(list(candidate))
                if len(short_term_tabu_list) > tabu_list_size:
                    short_term_tabu_list.pop(0)
                update_frequency_matrix(candidate)
                break  # Break after finding the first non-tabu solution

        if diversification_measure() > diversification_threshold:
            step_size *= intensification_factor
            frequency_matrix *= 0.5
        else:
            step_size *= (
                step_size_reduction_factor  # Apply the step size reduction factor
            )

    return best_solution, -best_value


# Example usage of the tabu_search function with predefined parameters
if __name__ == "__main__":
    individual = {
        "tabu_list_size": 169,
        "neighborhood_size": 54,
        "intensification_factor": 1.8505061762165624,
        "diversification_threshold": 0.6,
        "initial_step_size": 1,
        "step_size_reduction_factor": 0.99,
        "num_variables": 2,
    }

    solution, value = tabu_search(**individual)
    print(f"Best solution: {solution}")
    print(f"Best objective function value: {value}")
