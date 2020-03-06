import numpy as np
import sympy as sp
from numba import njit

x = sp.Symbol('x', Real=False)
y = sp.Symbol('y', Real=True)

symfunc1 = njit(sp.lambdify((x, y), 2*x*y, 'numpy'))
symfunc2 = njit(sp.lambdify((x, y), 4*x*x*y, 'numpy'))


@njit
def func():
    return np.arange(1000)


@njit
def main(argument):
    print(symfunc1(x=argument, y=argument))


if __name__ == "__main__":
    argument = np.arange(100)
    main(argument)
