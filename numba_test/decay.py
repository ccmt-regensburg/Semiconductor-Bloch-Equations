from numba import njit
from scipy.integrate import solve_ivp

@njit
def exponential_decay(t, y):
    return -0.5*y

sol = solve_ivp(exponential_decay, [0, 10], [2, 4, 8])

print(sol.t)
print(sol.y)
