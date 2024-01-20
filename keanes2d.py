import numpy as np
import matplotlib.pyplot as plt

def keanes_bump(x1, x2):
    return np.abs((np.cos(x1)**4 + np.cos(x2)**4 - 2*(np.cos(x1)**2)*(np.cos(x2)**2)) / np.sqrt(x1**2 + 2*x2**2))

def boundary_mask(x1, x2):
    return (0 <= x1) & (x1 <= 10) & (0 <= x2) & (x2 <= 10) & (x1*x2 > 0.75) & (x1 + x2 < 15)

x1 = np.linspace(0, 10, 400)
x2 = np.linspace(0, 10, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = keanes_bump(X1, X2)

mask = boundary_mask(X1, X2)
Z[~mask] = np.nan

fig, ax = plt.figure(figsize=(10, 8)), plt.axes()
contour = ax.contourf(X1, X2, Z, levels=50, cmap='viridis')

x1_boundary = np.linspace(0.75, 10, 400)
x2_boundary = 0.75 / x1_boundary
ax.plot(x1_boundary, x2_boundary, 'r--', label=r'$x_1x_2 > 0.75$')

x1_plus_x2 = np.linspace(0, 10, 400)
x2_plus_x2 = 15 - x1_plus_x2
ax.plot(x1_plus_x2, x2_plus_x2, 'b--', label=r'$x_1 + x_2 < 15$')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title("Keane's Bump Function Contour with Boundary Conditions")
ax.legend()
ax.set_ylim(0, 10)

fig.colorbar(contour, ax=ax, label='f(x1, x2)')
plt.show()
