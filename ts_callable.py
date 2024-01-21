import numpy as np
import matplotlib.pyplot as plt


def keanes_bump_function(x):
    # print(x)
    if (
        np.any(x < 0)
        or np.any(x > 10)
        or np.prod(x) <= 0.75
        or np.sum(x) >= 15 * len(x) / 2
    ):
        return np.inf  # Return infinity if constraints are violated
    sum_cos_x4 = np.sum(np.cos(x) ** 4)
    prod_cos_x2 = np.prod(np.cos(x) ** 2)
    indices = np.arange(1, len(x) + 1)
    numerator = sum_cos_x4 - 2 * prod_cos_x2
    denominator = np.sqrt(np.sum(indices * (x**2)))
    return -numerator / denominator  # Negative because we need to maximize


def local_search(base_point, delta, tabu_list, evaluate_function):
    best_value = evaluate_function(base_point)
    best_point = base_point.copy()

    # Explore the neighborhood
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

    # Pattern move (if there was an improvement)
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


# def update_medium_term_memory(
#     archive, candidate, candidate_value, archive_size, D_min=20, D_sim=2
# ):
#     def euclidean_distance(xa, xb):
#         return np.sqrt(np.sum((xa - xb) ** 2))

#     # If the archive is not full, add the candidate solution
#     if len(archive) < archive_size:
#         archive.append((candidate, candidate_value))
#         return

#     # Calculate the distances of the candidate to all solutions in the archive
#     distances = np.array([euclidean_distance(candidate, x[0]) for x in archive])

#     # Find the most similar and the worst solution in the archive
#     most_similar_index = np.argmin(distances)
#     worst_index = np.argmax([x[1] for x in archive])

#     # If the candidate is sufficiently dissimilar from all archived solutions or better than the worst
#     if (distances > D_min).all():
#         # Add candidate to the archive if it is not similar to any in the archive
#         archive.append((candidate, candidate_value))
#         # If the archive exceeds its size limit, remove the worst solution
#         if len(archive) > archive_size:
#             del archive[worst_index]
#     elif (candidate_value < archive[most_similar_index][1]) and (
#         distances[most_similar_index] < D_sim
#     ):
#         archive[most_similar_index] = (candidate, candidate_value)


def update_medium_term_memory(memory, point, value, size_limit, D_min=20, D_sim=2):
    """
    Update the medium-term memory with new solutions, keeping only the best ones.
    """
    # Check if the memory is not full or the new value is better than the worst in memory
    if len(memory) < size_limit or value < memory[-1][1]:
        # Insert the new solution and sort the memory based on values
        memory.append((point, value))
        memory.sort(key=lambda x: x[1])  # Sort by the objective function value
        # Keep only the best solutions if the memory exceeds its size limit
        while len(memory) > size_limit:
            memory.pop()  # Remove the worst solution


def tabu_search(
    tabu_list_size,
    medium_term_memory_size,
    initial_step_size,
    step_size_reduction_factor,
    num_variables=8,
    max_evaluations=10000,
    intensify_thresh=10,
    diversify_thresh=15,
    reduce_thresh=25,
    D_min=20,
    D_sim=2,
):
    current_search = np.random.uniform(0, 10, num_variables)
    current_search_sol = keanes_bump_function(current_search)
    evaluations = 1
    counter = 0
    short_term_memory = []
    medium_term_memory = []
    long_term_memory = np.zeros((num_variables, 10))

    step_size = initial_step_size

    while evaluations < max_evaluations:
        new_point = local_search(
            current_search, step_size, short_term_memory, keanes_bump_function
        )
        new_value = keanes_bump_function(new_point)
        evaluations += 2 * num_variables

        # Check if the new point is an improvement and not in the short-term memory
        if new_value < current_search_sol and list(new_point) not in short_term_memory:
            current_search, current_search_sol = new_point, new_value
            # Update medium-term memory with the new best solution
            update_medium_term_memory(
                medium_term_memory,
                new_point,
                new_value,
                medium_term_memory_size,
                D_min=D_min,
                D_sim=D_sim,
            )

            # Update long-term memory
            for i, x in enumerate(new_point):
                index = min(int(np.floor(x)), 9)  # Ensure index is within bounds [0, 9]
                long_term_memory[i, index] += 1
        else:
            counter += 1

        # Add the new point to the short-term memory
        short_term_memory.append(list(new_point))
        if len(short_term_memory) > tabu_list_size:
            short_term_memory.pop(0)

        # Intensification: Move to the average of the best M solutions
        if counter >= intensify_thresh:
            # print("INTENSIFY")
            best_solutions = np.array([m[0] for m in medium_term_memory])
            current_search = np.mean(best_solutions, axis=0)
            # new_best_solution_counter = 0
        if counter >= diversify_thresh:
            # print("DIVERSIFY")
            # Identify the least visited quadrant
            least_visited_indices = np.argmin(long_term_memory, axis=1)
            # Randomly select a point within that quadrant
            for i, idx in enumerate(least_visited_indices):
                range_min = idx  # assuming each quadrant is 1 unit
                range_max = min(idx + 1, 9)  # prevent index out of bounds
                current_search[i] = np.random.uniform(range_min, range_max)
            current_search = np.clip(current_search, 0, 10)
        if counter >= reduce_thresh:
            # print("REDUCE")
            step_size *= step_size_reduction_factor
            counter = 0
        if step_size < initial_step_size / 10000:
            # print(f"Step size: {step_size}")
            return medium_term_memory
    # print(f"Step size: {step_size}")
    return medium_term_memory


def keanes_bump(x1, x2):
    return np.abs(
        (np.cos(x1) ** 4 + np.cos(x2) ** 4 - 2 * (np.cos(x1) ** 2) * (np.cos(x2) ** 2))
        / np.sqrt(x1**2 + 2 * x2**2)
    )


def boundary_mask(x1, x2):
    return (
        (0 <= x1)
        & (x1 <= 10)
        & (0 <= x2)
        & (x2 <= 10)
        & (x1 * x2 > 0.75)
        & (x1 + x2 < 15)
    )


def plot_2Dkeanes_bump_with_memory(medium_term_memory):
    x1 = np.linspace(0, 10, 400)
    x2 = np.linspace(0, 10, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = keanes_bump(X1, X2)

    mask = boundary_mask(X1, X2)
    Z[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X1, X2, Z, levels=50, cmap="viridis")

    # Plot the medium term memory points
    mtm_x1, mtm_x2 = zip(*[point for point, value in medium_term_memory])
    ax.plot(mtm_x1, mtm_x2, "ro", markersize=5, label="Medium Term Memory")

    x1_boundary = np.linspace(0.75, 10, 400)
    x2_boundary = 0.75 / x1_boundary
    ax.plot(x1_boundary, x2_boundary, "r--", label=r"$x_1x_2 > 0.75$")

    x1_plus_x2 = np.linspace(0, 10, 400)
    x2_plus_x2 = 15 - x1_plus_x2
    ax.plot(x1_plus_x2, x2_plus_x2, "b--", label=r"$x_1 + x_2 < 15$")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Keane's Bump Function Contour with Boundary Conditions")
    ax.legend()
    ax.set_ylim(0, 10)

    fig.colorbar(contour, ax=ax, label="f(x1, x2)")
    plt.show()


if __name__ == "__main__":
    individual = {
        "num_variables": 8,
        "initial_step_size": 4,
        "step_size_reduction_factor": 0.8,
        "tabu_list_size": 7,
        "medium_term_memory_size": 10,
        "D_min": 20,
        "D_sim": 2,
        "intensify_thresh": 10,
        "diversify_thresh": 15,
        "reduce_thresh": 25,
    }
    individual = {
        "num_variables": 8,
        "initial_step_size": 3.8377312562936727,
        "step_size_reduction_factor": 0.46319761829933004,
        "tabu_list_size": 10,
        "medium_term_memory_size": 22,
        "D_min": 94.57832785852996,
        "D_sim": 7.787363437552019,
        "intensify_thresh": 10,
        "diversify_thresh": 15,
        "reduce_thresh": 25,
    }
    archive = tabu_search(**individual)
    print(archive)
    best_index = np.argmin([x[1] for x in archive])
    solution, value = (
        archive[best_index][0],
        -archive[best_index][1],
    )
    # print(np.prod(solution), np.sum(solution)<15*4)
    print(f"Best solution: {solution}")
    print(f"Best objective function value: {value}")
    if individual["num_variables"] == 2:
        plot_2Dkeanes_bump_with_memory(archive)
