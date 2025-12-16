import sympy as sp

def Lagrange_Equality(obj_fn, constraints, n_vars=1, minimize=True):
    sp_n_vars = sp.symbols(' '.join([f'x_{i}' for i in range(n_vars)]))
    lambdas = sp.symbols(' '.join([f'lambda_{i}' for i in range(len(constraints))]))
    if len(constraints) == 1:
        lambdas = (lambdas,)

    sp_obj_fn = sp.sympify(obj_fn, locals={f'x_{i}': sp_n_vars[i] for i in range(n_vars)})
    sp_constraints = [sp.sympify(c, locals={f'x_{i}': sp_n_vars[i] for i in range(n_vars)}) for c in constraints]
    sp_lagrangian = sp_obj_fn + sum(lambdas[i] * sp_constraints[i] for i in range(len(sp_constraints)))

    eqs = [sp.diff(sp_lagrangian, x) for x in sp_n_vars] + [sp.diff(sp_lagrangian, l) for l in lambdas]

    solution = sp.solve(eqs, sp_n_vars + lambdas, dict=True)
    solution = solution[0]

    best_point = [float(solution[x].evalf()) for x in sp_n_vars]
    best_value = float(sp_obj_fn.subs(solution).evalf())
    best_value = round(best_value, 4)
    best_point = [round(p, 4) for p in best_point]

    return best_point, best_value

print('1. Max 2𝑥 +  𝑦   + 10\n   s.t  𝑥 + 2𝑦^2 = 3')
print(Lagrange_Equality('2*x_0 + x_1 + 10', ['x_0 + 2*x_1**2 - 3'], 2, minimize=False))

print('\n\n2. Min 𝑥^2 + 𝑦^2 + 𝑧^2\n   s.t 𝑥   + 2𝑦  + 3𝑧  = 7  2𝑥 + 2𝑦 + 𝑧 = 9/2')
print(Lagrange_Equality('x_0**2 + x_1**2 + x_2**2',
                        ['x_0 + 2*x_1 + 3*x_2 - 7', '2*x_0 + 2*x_1 + x_2 - 4.5'],
                        3, minimize=True))
