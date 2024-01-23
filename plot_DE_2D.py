import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution


def keanes_bump(x):
    x1, x2 = x.T
    valid = (
        (0 <= x1)
        & (x1 <= 10)
        & (0 <= x2)
        & (x2 <= 10)
        & (x1 * x2 > 0.75)
        & (x1 + x2 < 15)
    )
    result = np.full(x1.shape, np.inf)
    result[valid] = -np.abs(
        (
            np.cos(x1[valid]) ** 4
            + np.cos(x2[valid]) ** 4
            - 2 * (np.cos(x1[valid]) ** 2) * (np.cos(x2[valid]) ** 2)
        )
        / np.sqrt(x1[valid] ** 2 + 2 * x2[valid] ** 2)
    )
    return result


bounds = [(0, 10), (0, 10)]
best_solutions = []


def callback(x, convergence):
    best_solutions.append(np.copy(x))


popsize = 5
maxiter = round(10000 / popsize / len(bounds) - 1)
recombination = 0.3
mutation = 0.8
strategy_params = {
    "strategy": "best1bin",
    "recombination": recombination,
    "mutation": mutation,
    "maxiter": maxiter,
    "popsize": popsize,
    "tol": 0.001,
    "updating": "deferred",
    "callback": callback,
}
result = differential_evolution(
    keanes_bump, bounds, **strategy_params, disp=True
)

x1 = np.linspace(0, 10, 400)
x2 = np.linspace(0, 10, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = keanes_bump(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(X1, X2, -Z, levels=50, cmap="viridis")
fig.colorbar(contour, ax=ax, label="f(x1, x2)")

best_solutions = np.array(best_solutions)
ax.plot(best_solutions[:, 0], best_solutions[:, 1], "r.-", label="Best Solution Path")
ax.scatter(
    best_solutions[-1, 0], best_solutions[-1, 1], color="green", label="Final Solution"
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title("Keane's Bump Function with Optimization Path")
ax.legend()

plt.show()
