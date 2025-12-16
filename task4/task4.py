import sympy as sp
import numpy as np

def Steepest(obj_fn, initial_guess, n_vars=1, descent=True, epochs=10):
    """
    Implements Steepest Descent/Ascent method with optimized learning rate.

    Parameters:
        obj_fn (string): Objective function. In variables x_0, x_1, and so on
        initial_guess (list of floats): Initial point from which the iterative method begins.
        n_vars (int): Number of variables, default is 1
        descent (boolean): True for minimization problems, False for maximization problems
        epochs (int): number of iterations

    Returns:
        tuple: ([best_point], best_value) rounded to 4 decimal points
    """
    #define variables
    vars_sym = sp.symbols(' '.join([f'x_{i}' for i in range(n_vars)]))

    #convert function
    f = sp.sympify(obj_fn)

    #get gradient
    grad_f = [sp.diff(f, v) for v in vars_sym]

    #current point
    x_curr = np.array(initial_guess, dtype=float)

    for _ in range(epochs):
        #gradient @ current point
        grad_val = np.array([float(g.subs({vars_sym[i]: x_curr[i] for i in range(n_vars)}))
                             for i, g in enumerate(grad_f)])

        #gradient direction
        direction = -grad_val if descent else grad_val

        # Line search to find optimal learning rate (alpha)
        alpha = sp.symbols('alpha')
        x_next_sym = [x_curr[i] + alpha * direction[i] for i in range(n_vars)]
        f_alpha = f.subs({vars_sym[i]: x_next_sym[i] for i in range(n_vars)})
        alpha_opt = sp.solve(sp.diff(f_alpha, alpha), alpha)

        #alpha
        alpha_val = 0.01
        for val in alpha_opt:
            if val.is_real:
                alpha_val = float(val)
                break

        #update current point
        x_curr = x_curr + alpha_val * direction

    best_point = [float(round(val, 4)) for val in x_curr]
    best_value = round(float(f.subs({vars_sym[i]: x_curr[i] for i in range(n_vars)})), 4)

    return best_point, best_value

#Min x - y + 2x^2 + 2xy + y^2, x0=0, y0=0, 2 iterations
result1 = Steepest("x_0 - x_1 + 2*x_0**2 + 2*x_0*x_1 + x_1**2", [0, 0], n_vars=2, descent=True, epochs=2)
print(result1)

#Min 6x^2 - 6xy + 2y^2 - x - 2y, x0=0, y0=0, 4 iterations
result2 = Steepest("6*x_0**2 - 6*x_0*x_1 + 2*x_1**2 - x_0 - 2*x_1", [0, 0], n_vars=2, descent=True, epochs=4)
print(result2)
