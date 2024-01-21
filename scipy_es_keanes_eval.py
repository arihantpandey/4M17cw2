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

# Define bounds for each of the 8 parameters
bounds = [(0, 10)] * 8

# Define ranges for recombination and mutation parameters
recombination_range = np.linspace(0, 1, 15)
mutation_range = np.linspace(0, 2, 15, endpoint=False)

# Record the performance
performance_data = []
final_values = np.zeros((len(recombination_range), len(mutation_range)))
number_iters = np.zeros((len(recombination_range), len(mutation_range)))

for i, recombination in enumerate(recombination_range):
    for j, mutation in enumerate(mutation_range):
        popsize = 50
        maxiter = round(10000/popsize/len(bounds) - 1)
        strategy_params = {
            'strategy': 'best1bin',
            'recombination': recombination,
            'mutation': mutation,
            'maxiter': maxiter,
            'popsize': popsize,
            'tol': 0.001,
            'updating': 'deferred'
        }

        result = differential_evolution(keanes_bump, bounds, **strategy_params)
        final_values[i, j] = -result.fun
        number_iters[i,j] = (result.nit+1)*popsize*len(bounds)
        performance_data.append({
            'recombination': recombination,
            'mutation': mutation,
            'fun': -result.fun,
            'iterations': result.nit
        })
        print(performance_data[-1])

# Print results
for data in performance_data:
    print(f"Recombination: {data['recombination']}, Mutation: {data['mutation']}, "
          f"Final value: {data['fun']}, Iterations: {data['iterations']}")

def format_func2(value, tick_number):
    return f'{value:.3f}'   
plt.figure(figsize=(20, 16))
sns.heatmap(final_values, xticklabels=np.round(mutation_range, 2), yticklabels=np.round(recombination_range, 2), annot=True, fmt=".3f", cmap="viridis")
plt.title('Heatmap of Final Function Value')
plt.xlabel('Mutation')
plt.ylabel('Recombination')

# Format the numbers to avoid scientific notation
def format_func(value, tick_number):
    return f'{value:.0f}'

plt.figure(figsize=(20, 16))
ax = sns.heatmap(number_iters, xticklabels=np.round(mutation_range, 2), yticklabels=np.round(recombination_range, 2), annot=True, fmt=".0f", cmap="viridis")
ax.collections[0].colorbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.title('Heatmap of Number of Obj Func Evals')
plt.xlabel('Mutation')
plt.ylabel('Recombination')

plt.show()