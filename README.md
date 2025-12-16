# Unified Optimization Function

This project implements a **unified optimization utility** in Python that supports **symbolic and numerical optimization methods** using **SymPy** and **NumPy**. The function is designed for educational and practical use, demonstrating multiple classical optimization techniques in a single, flexible interface.

---

## Features

The `optimize` function supports the following optimization methods:

1. **Calculus-Based Optimization**

   * Finds critical points symbolically using gradients.
   * Suitable for **unconstrained** problems.

2. **Lagrange Multipliers**

   * Solves **constrained optimization** problems with equality constraints.
   * Uses symbolic differentiation and equation solving.

3. **Newton's Method**

   * Numerical optimization using gradient and Hessian matrices.
   * Efficient for smooth, unconstrained problems.

4. **Steepest Descent / Ascent**

   * Iterative gradient-based optimization.
   * Automatically computes the optimal step size using symbolic line search.

---

## Requirements

* Python 3.x
* NumPy
* SymPy

Install dependencies with:

```bash
pip install numpy sympy
```

---

## Function Overview

```python
optimize(obj_fn, method='calculus_based_opt', n_vars=1, minimize=True,
         constraints=None, inequality=False,
         initial_guess=None, epsilon=0.001,
         descent=True, epochs=10)
```

### Parameters

* **obj_fn (str)**: Objective function written in terms of `x_0, x_1, ...`
* **method (str)**: Optimization method

  * `'calculus_based_opt'`
  * `'lagrange'`
  * `'newton'`
  * `'steepest'`
* **n_vars (int)**: Number of variables
* **minimize (bool)**: Minimize if `True`, maximize if `False`
* **constraints (list[str])**: Constraints for Lagrange method
* **inequality (bool)**: Placeholder for inequality constraints (not implemented)
* **initial_guess (list[float])**: Initial guess for Newton and Steepest methods
* **epsilon (float)**: Convergence threshold for Newton's method
* **descent (bool)**: Descent (`True`) or ascent (`False`) for steepest method
* **epochs (int)**: Number of iterations for steepest method

### Returns

* **tuple**: `([best_point], best_value)` rounded to 4 decimal places

---

## Completed Tasks & Examples

### 1. Calculus-Based Optimization

Finds symbolic critical points and evaluates them.

```python
optimize("x_0**3 - x_0", method='calculus_based_opt', n_vars=1, minimize=True)
```

**Result:**

```
([0.5774], -0.3849)
```

---

### 2. Constrained Optimization (Lagrange Multipliers)

Solves constrained problems using symbolic equations.

```python
optimize('2 * x_0 + x_1 + 10', method='lagrange', n_vars=2,
         minimize=False, constraints=['x_0 + 2 * x_1**2 - 3'])
```

**Result:**

```
([2.9688, 0.125], 16.0625)
```

---

### 3. Newton's Method

Numerical optimization using gradient and Hessian matrices.

```python
optimize('2 * sin(x_0) - 0.1 * x_0**2', method='newton',
         n_vars=1, initial_guess=[2.5], epsilon=0.05)
```

**Result:**

```
([1.4276], 1.7757)
```

---

### 4. Steepest Descent Method

Iterative optimization with symbolic line search.

```python
optimize("x_0 - x_1 + 2 * x_0**2 + 2 * x_0 * x_1 + x_1**2",
         method='steepest', n_vars=2,
         initial_guess=[0, 0], descent=True, epochs=2)
```

**Result:**

```
([-0.8, 1.2], -1.2)
```

---
