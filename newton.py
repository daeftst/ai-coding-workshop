# ABOUTME: Univariate Newton's method for optimization using finite difference derivatives.
# ABOUTME: Finds a local minimum of f(x) by iteratively applying the Newton update rule.

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
