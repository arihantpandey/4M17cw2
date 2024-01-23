from scipy.optimize import differential_evolution
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Set the bounds for the Keane's Bump function
bounds = [(0, 10)] * 8
popsize = 5
maxiter = round(10000/popsize/len(bounds) - 1)

# Define all strategy variants
strategies = ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin', 'rand1bin']

# Prepare the figure
plt.figure(figsize=(8.3, 11.7))  # A4 size in inches
num_evaluations = 100

for i, strategy in enumerate(strategies, 1):
    strategy_params = {
        'strategy': strategy,
        'recombination': 0.3,
        'mutation': 0.8,
        'maxiter': maxiter,
        'popsize': popsize,
        'tol': 0.001,
        'updating': 'deferred'
    }

    final_values = []

    for _ in range(num_evaluations):
        result = differential_evolution(keanes_bump, bounds, **strategy_params)
        final_values.append(-result.fun)

    ax = plt.subplot(4, 3, i)
    sns.violinplot(final_values)
    ax.set_title(f'{strategy}')
    ax.set_xlabel('Final Value')
    ax.set_ylabel('Density')
    ax.set_xlim(0.4, 0.8)  # Set x-axis limits

plt.tight_layout()
plt.show()
