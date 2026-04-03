# ABOUTME: Newton's method for optimization (univariate and multivariate) using finite difference derivatives.
# ABOUTME: Finds a local minimum by iteratively applying the Newton update rule.

import numpy as np


def optimize(f, x0, tol=1e-6, max_iter=100):
    h = 1e-5
    second_deriv_eps = 1e-10
    x = x0

    for _ in range(max_iter):
        fx = f(x)
        f_plus = f(x + h)
        f_minus = f(x - h)

        first_deriv = (f_plus - f_minus) / (2 * h)
        second_deriv = (f_plus - 2 * fx + f_minus) / (h ** 2)

        if abs(second_deriv) < second_deriv_eps:
            return {'x': x, 'converged': False}

        x_new = x - first_deriv / second_deriv

        if abs(x_new - x) < tol:
            return {'x': x_new, 'converged': True}

        x = x_new

    return {'x': x, 'converged': False}


def optimize_nd(f, x0, tol=1e-6, max_iter=100):
    h = 1e-5
    x = np.array(x0, dtype=float)
    n = len(x)

    for _ in range(max_iter):
        fx = f(x)

        # Gradient via central finite differences
        grad = np.zeros(n)
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = h
            grad[i] = (f(x + e_i) - f(x - e_i)) / (2 * h)

        # Hessian via finite differences
        hess = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = h
            hess[i, i] = (f(x + e_i) - 2 * fx + f(x - e_i)) / (h ** 2)
            for j in range(i + 1, n):
                e_j = np.zeros(n)
                e_j[j] = h
                val = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * h ** 2)
                hess[i, j] = val
                hess[j, i] = val

        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return {'x': x, 'converged': False}

        x_new = x - delta

        if np.linalg.norm(x_new - x) < tol:
            return {'x': x_new, 'converged': True}

        x = x_new

    return {'x': x, 'converged': False}
