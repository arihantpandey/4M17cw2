import numpy as np

def keanes_bump(x):
    x = np.array(x)
    if np.any(x <= 0) or np.any(x >= 10) or np.prod(x) <= 0.75 or np.sum(x) >= (15 * len(x)) / 2:
        return -np.inf
    
    term1 = np.sum(np.cos(x)**4)
    term2 = 2 * np.prod(np.cos(x)**2)
    indices = np.arange(1, len(x) + 1)
    term3 = np.sqrt(np.sum(indices * (x**2)))
    result = np.abs((term1 - term2) / term3)
    
    return result

x_test = np.random.uniform(0.75, 10, 8)
keanes_bump_value = keanes_bump(x_test)
