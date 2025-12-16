import numpy as np
import sympy as sp

def optimize(obj_fn, method='calculus_based_opt', n_vars=1, minimize=True,
             constraints=None, inequality=False,
             initial_guess=None, epsilon=0.001,
             descent=True, epochs=10):
    """
    Unified optimization function that supports multiple optimization methods.

    Parameters:
        obj_fn (str): Objective function in variables x_0, x_1, and so on.
                     Example: "(x_0 - 2)**2 + (x_1 - 3)**2"
        method (str): Optimization method to use. Options:
                     - 'calculus_based_opt': Symbolic optimization using gradient and Hessian (unconstrained)
                     - 'lagrange': Lagrange method for constrained optimization
                     - 'newton': Newton's method for numerical unconstrained optimization
                     - 'steepest': Steepest descent/ascent method
        n_vars (int): Number of variables, default is 1
        minimize (bool): True for minimization, False for maximization, default is True
        constraints (list of str, optional): Constraint(s) for Lagrange method.
                                            Example: ['2 * x_0 + x_1 - 4']
        inequality (bool): True for inequality constraints, False for equality constraints (Lagrange only)
        initial_guess (list of floats, optional): Initial point for Newton and Steepest methods
        epsilon (float): Error threshold for Newton method, default is 0.001
        descent (bool): True for minimization, False for maximization (Steepest only), default is True
        epochs (int): Number of iterations for Steepest method, default is 10

    Returns:
        tuple: ([best_point], best_value) rounded to 4 decimal points
    """

    vars_ = sp.symbols([f'x_{i}' for i in range(n_vars)])
    obj = sp.sympify(obj_fn)

    best_point = []
    best_value = 0.0

    if method == 'calculus_based_opt':
        grad = [sp.diff(obj, v) for v in vars_]
        critical_points = sp.solve(grad, vars_, dict=True)

        candidates = []
        for point in critical_points:
            val = obj.subs(point)
            p_coords = [float(point[v]) for v in vars_]
            candidates.append((p_coords, float(val)))

        candidates.sort(key=lambda x: x[1], reverse=not minimize)
        best_point, best_value = candidates[0]

    elif method == 'lagrange':
        lam = sp.symbols('lambda')
        constraint_expr = sp.sympify(constraints[0])

        lagrangian = obj + lam * constraint_expr
        eqs = [sp.diff(lagrangian, v) for v in vars_] + [constraint_expr]

        solutions = sp.solve(eqs, vars_ + [lam], dict=True)

        candidates = []
        for sol in solutions:
            pt = [float(sol[v]) for v in vars_]
            val = float(obj.subs(sol))
            candidates.append((pt, val))

        candidates.sort(key=lambda x: x[1], reverse=not minimize)
        best_point, best_value = candidates[0]

    elif method == 'newton':
        grad = [sp.diff(obj, v) for v in vars_]
        x = np.array(initial_guess, dtype=float)

        grad_func = sp.lambdify(vars_, grad, 'numpy')
        hessian = sp.hessian(obj, vars_)
        hess_func = sp.lambdify(vars_, hessian, 'numpy')

        while True:
            g = np.array(grad_func(*x), dtype=float).flatten()
            if np.linalg.norm(g) < epsilon:
                break

            H = np.array(hess_func(*x), dtype=float)
            if n_vars == 1: H = H.reshape(1, 1)

            x = x - np.linalg.inv(H) @ g

        best_point = [float(v) for v in x]
        best_value = float(obj.subs(dict(zip(vars_, x))))

    elif method == 'steepest':
        grad = [sp.diff(obj, v) for v in vars_]
        x = np.array(initial_guess, dtype=float)
        grad_func = sp.lambdify(vars_, grad, 'numpy')
        alpha_sym = sp.symbols('alpha_sym')

        for _ in range(epochs):
            g = np.array(grad_func(*x), dtype=float).flatten()

            if descent:
                x_next_expr = x - alpha_sym * g
            else:
                x_next_expr = x + alpha_sym * g

            subs_dict = dict(zip(vars_, x_next_expr))
            f_alpha = obj.subs(subs_dict)

            diff_alpha = sp.diff(f_alpha, alpha_sym)
            optimal_alpha = float(sp.solve(diff_alpha, alpha_sym)[0])

            if descent:
                x = x - optimal_alpha * g
            else:
                x = x + optimal_alpha * g

        best_point = [float(v) for v in x]
        best_value = float(obj.subs(dict(zip(vars_, x))))

    best_point = [round(x, 4) for x in best_point]
    best_value = round(best_value, 4)

    return best_point, best_value

# Example 1: Calculus Based Optimization
print(optimize("x_0**3 - x_0", method='calculus_based_opt', n_vars=1, minimize=True))
# Expected: ([0.5774], -0.3849)

# Example 2: Lagrange Multipliers
print(optimize('2 * x_0 + x_1 + 10', method='lagrange', n_vars=2, minimize=False, constraints=['x_0 + 2 * x_1**2 - 3']))
# Expected: ([2.9688, 0.125], 16.0625)

# Example 3: Newton's Method
print(optimize('2 * sin(x_0) - 0.1 * x_0**2', method='newton', n_vars=1, initial_guess=[2.5], epsilon=0.05))
# Expected: ([1.4276], 1.7757)

# Example 4: Steepest Descent
print(optimize("x_0 - x_1 + 2 * x_0**2 + 2 * x_0 * x_1 + x_1**2", method='steepest', n_vars=2, initial_guess=[0, 0], descent=True, epochs=2))
# Expected: ([-0.8, 1.2], -1.2)
