import sympy as sp

def Unconstrained_Optimizer(fn, n_vars=1, minimize=True):
    xs = sp.symbols(' '.join(f'x_{i}' for i in range(n_vars)))
    if n_vars == 1:
        xs = (xs,)

    environment = {str(s): s for s in xs}
    symbolic_objective = sp.sympify(fn, locals=environment)

    gradient = [sp.diff(symbolic_objective, s) for s in xs]
    critical_points = sp.solve(gradient, xs, dict=True)
    hessian_matrix = sp.hessian(symbolic_objective, xs)

    optimal_point = None
    optimal_value = sp.oo if minimize else -sp.oo

    for pt in critical_points:
        if any(v.is_real is False for v in pt.values()):
            continue

        H_at = hessian_matrix.subs(pt)

        try:
            eigs = [float(ev) for ev in H_at.eigenvals().keys()]
        except Exception:
            continue

        is_min = all(ev > 0 for ev in eigs)
        is_max = all(ev < 0 for ev in eigs)

        if (minimize and not is_min) or ((not minimize) and not is_max):
            continue

        val_here = float(symbolic_objective.subs(pt))

        if (minimize and val_here < float(optimal_value)) or ((not minimize) and val_here > float(optimal_value)):
            optimal_value = val_here
            optimal_point = [float(pt[s]) for s in xs]


    if optimal_point is None:
        best_point, optimal_value = None, None
    else:
        best_point = round(optimal_point[0], 4) if n_vars == 1 else [round(u, 4) for u in optimal_point]
        best_value = round(float(optimal_value), 4)

    return best_point, best_value


print("Question #1: Min 𝑥3−𝑥")
print("---------------------")
print(Unconstrained_Optimizer("x_0**3 - x_0", n_vars=1, minimize=True))
print("\n______________________________________")
print("______________________________________\n")
print("Question #2: Max 20𝑥+26𝑦+4𝑥𝑦−4𝑥2−3𝑦2")
print("---------------------")
print(Unconstrained_Optimizer("20*x_0 + 26*x_1 + 4*x_0*x_1 - 4*x_0**2 - 3*x_1**2", n_vars=2, minimize=False))
