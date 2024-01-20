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


def local_search(base_point, delta, tabu_list, evaluate_function):
    best_value = evaluate_function(base_point)
    best_point = base_point.copy()

    # Step 3: Explore the neighborhood
    for i in range(len(base_point)):
        for d in [-delta, delta]:  # Decrease and increase the current variable
            new_point = best_point.copy()
            new_point[i] += d
            if 0 <= new_point[i] <= 10:  # Check variable bounds
                if list(new_point) not in tabu_list:
                    new_value = evaluate_function(new_point)
                    if new_value < best_value:  # Check if this is a better solution
                        best_value = new_value
                        best_point = new_point

    # Step 5: Pattern move (if there was an improvement)
    if not np.array_equal(best_point, base_point):
        pattern_point = base_point + (best_point - base_point)
        if (
            all(0 <= pattern_point)
            and all(pattern_point <= 10)
            and list(pattern_point) not in tabu_list
        ):  # Check bounds and tabu
            pattern_value = evaluate_function(pattern_point)
            if pattern_value < best_value:
                return pattern_point  # Return the pattern move if it's better
            else:
                return best_point  # Otherwise, return the single-step improvement
        else:
            return best_point  # Return the single-step improvement if pattern move is not allowed
    else:
        return base_point  # Return the base point if no improvement was found


def tabu_search(
    tabu_list_size,
    initial_step_size,
    step_size_reduction_factor,
    medium_term_memory_size,
    num_variables=8,
    max_evaluations=10000,
    intensify_thresh=10,
    diversify_thresh=15,
    reduce_thresh=25,
):
    best_solution = np.random.uniform(0, 10, num_variables)
    # best_solution = np.array([
    #     2.10560965,
    #     3.18194561,
    #     0.87809551,
    #     3.22561506,
    #     2.79578934,
    #     0.72160917,
    #     0.22399556,
    #     0.19429642,
    # ])
    best_value = keanes_bump_function(best_solution)
    evaluations = 1
    counter = 0
    short_term_memory = []
    medium_term_memory = []
    long_term_memory = np.zeros(
        (num_variables, 10)
    )  # For discretizing the search space

    step_size = initial_step_size

    while evaluations < max_evaluations:
        new_point = local_search(
            best_solution, step_size, short_term_memory, keanes_bump_function
        )
        new_value = keanes_bump_function(new_point)
        evaluations += 2 * num_variables

        # Check if the new point is an improvement and not in the short-term memory
        if new_value < best_value and list(new_point) not in short_term_memory:
            best_solution, best_value = new_point, new_value
            # Update medium-term memory with the new best solution
            medium_term_memory.append((new_point, new_value))
            if len(medium_term_memory) > medium_term_memory_size:
                medium_term_memory.pop(0)

            # Update long-term memory
            for i, x in enumerate(new_point):
                index = min(int(np.floor(x)), 9)  # Ensure index is within bounds [0, 9]
                long_term_memory[i, index] += 1
        else:
            counter+=1

        # Add the new point to the short-term memory
        short_term_memory.append(list(new_point))
        if len(short_term_memory) > tabu_list_size:
            short_term_memory.pop(0)

        # Intensification: Move to the average of the best M solutions
        if counter >= intensify_thresh:
            print("INTENSIFY")
            best_solutions = np.array([m[0] for m in medium_term_memory])
            best_solution = np.mean(best_solutions, axis=0)
            # new_best_solution_counter = 0

        # Diversification: Find the least visited area in the long-term memory
        if counter >= diversify_thresh:
            print("DIVERSIFY")
            least_visited_indices = np.argmin(long_term_memory, axis=1)
            best_solution = least_visited_indices + 0.5
            best_solution = np.clip(best_solution, 0, 9.999)
            # step_size = initial_step_size  # Reset the step size
            # new_best_solution_counter = 0

        if counter >= reduce_thresh:
            print("REDUCE")
            step_size *= step_size_reduction_factor
            counter = 0
        if step_size < 0.01:
            return best_solution, -best_value
    print(f"Step size: {step_size}")
    return best_solution, -best_value


# Example usage of the tabu_search function with predefined parameters
if __name__ == "__main__":
    individual = {
        "tabu_list_size": 7,
        "initial_step_size": 2,
        "step_size_reduction_factor": 0.8,
        "medium_term_memory_size": 4,  # Example value for M
    }
   
    solution, value = tabu_search(**individual)
    print(np.prod(solution), np.sum(solution)<15*4)
    print(f"Best solution: {solution}")
    print(f"Best objective function value: {value}")
