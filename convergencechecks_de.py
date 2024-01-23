from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt


# Define the Keane's Bump function
def keanes_bump(x):
    if (
        np.any(x <= 0)
        or np.any(x >= 10)
        or np.prod(x) <= 0.75
        or np.sum(x) >= (15 * len(x)) / 2
    ):
        return 0

    term1 = np.sum(np.cos(x) ** 4)
    term2 = 2 * np.prod(np.cos(x) ** 2)
    indices = np.arange(1, len(x) + 1)
    term3 = np.sqrt(np.sum(indices * (x**2)))
    result = np.abs((term1 - term2) / term3)

    return -result


# Set the bounds for the Keane's Bump function
bounds = [(0, 10)] * 8

# Define the parameter pairs for recombination and mutation
param_pairs = [(0.29, 0.43), (0.14, 0.85), (0.14, 1.85), (0.93, 0.71), (0.64, 0.85), (0.14,0.14)]

# Prepare the figure
plt.figure()

for i, (recombination, mutation) in enumerate(param_pairs, 1):
    # Callback function to capture the best objective value at each iteration
    iter_values = []

    callback = lambda xk, convergence: iter_values.append(keanes_bump(xk))

    popsize = 5
    maxiter = round(10000 / popsize / len(bounds) - 1)
    # Run differential evolution
    result = differential_evolution(
        keanes_bump,
        bounds,
        strategy="best1bin",
        recombination=recombination,
        mutation=mutation,
        callback=callback,
        maxiter=maxiter,
        popsize=popsize,
    )

    # Plotting convergence for this pair
    plt.plot(iter_values, label=f"Recombination: {recombination}, Mutation: {mutation}")
plt.xlabel("Iterations")
plt.ylabel("Objective Value")
plt.legend()
plt.title(
    f"Convergence for varying Recombination and Mutation for popsize=5"
)

plt.tight_layout()
plt.show()
