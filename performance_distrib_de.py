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
popsize = 10
maxiter = round(10000/popsize/len(bounds) - 1)
# Set the strategy parameters
strategy_params = {
    'strategy': 'best1bin',
    'recombination': 0.3,
    'mutation': 0.8,
    'maxiter': maxiter,
    'popsize': popsize,
    'tol': 0.001,
    'updating': 'deferred'
}

# Number of evaluations
num_evaluations = 100

# Lists for storing results
final_values = []
number_iterations = []

# Run the optimization 100 times
for _ in range(num_evaluations):
    result = differential_evolution(keanes_bump, bounds, **strategy_params)
    final_values.append(-result.fun)
    number_iterations.append(result.nit)

# Box and violin plots for final values
plt.figure()
sns.violinplot(final_values)
plt.title('Violin Plot of Final Values')

plt.tight_layout()

# Box and violin plots for number of iterations
plt.figure()
sns.violinplot((np.array(number_iterations)+1)*popsize*len(bounds))
plt.title('Violin Plot of Number of Obj Func Evals')

plt.tight_layout()
plt.show()
