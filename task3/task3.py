import numpy as np
import sympy as sp

def Newton(obj_fn, initial_guess, n_vars=1, epsilon=0.001):
    """
    Implements Newton method to numerically solve univariate and multivariate unconstrained optimization problems.
    Parameters:
        obj_fn (string): Objective function. In variables x_0, x_1, and so on
        initial_guess (list of floats): Initial point from which the iterative method begins.
        n_vars (int): Number of variables, default is 1
        epsilon (float): Error threshold for the stopping condition based on the **absolute error** between successive iterates |x_i+1 - x_i|

    Returns:
        tuple: ([best_point], best_value) rounded to 4 decimal points
    """

    #symbolic variables
    vars_list = [sp.symbols(f'x_{i}') for i in range(n_vars)]

    #function to symbolic expression
    f = sp.sympify(obj_fn)

    #gradient
    gradient = [sp.diff(f, var) for var in vars_list]

    #Hessian
    hessian = [[sp.diff(gradient[i], vars_list[j]) for j in range(n_vars)] for i in range(n_vars)]

    #symbolic expression to numerical functions
    grad_fn = sp.lambdify(vars_list, gradient, "numpy")
    hess_fn = sp.lambdify(vars_list, hessian, "numpy")
    f_fn = sp.lambdify(vars_list, f, "numpy")

    #first guess to numpy array
    x_current = np.array(initial_guess, dtype=float)

    while True:
        grad_val = np.array(grad_fn(*x_current), dtype=float)
        hess_val = np.array(hess_fn(*x_current), dtype=float)

        #H*p
        p = np.linalg.solve(hess_val, -grad_val)

        #update p
        x_next = x_current + p

        #stopping condition
        if np.linalg.norm(x_next - x_current) < epsilon:
            break

        x_current = x_next

    #round
    best_point = [round(float(x), 4) for x in x_next]
    best_value = round(float(f_fn(*x_next)), 4)

    return best_point, best_value

Newton('x_0 - x_1 + 2 * x_0**2 + 2 * x_0 * x_1 + x_1**2', [0, 0], n_vars=2, epsilon=0.001)

Newton('2 * sin(x_0) - 0.1 * x_0**2', [2.5], n_vars=1, epsilon=0.05)
