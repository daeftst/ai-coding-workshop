# Newton's Method for Optimization

A hands-on exercise: implement Newton's method for univariate optimization using Claude Code.

## Background

Newton's method for optimization finds a local minimum of a function $f(x)$ by iteratively refining an estimate using second-order information. Starting from an initial guess $x_0$, the update rule is:

$$x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}$$

This differs from the root-finding version of Newton's method. Here, we're looking for where the derivative is zero (a stationary point), using the second derivative to determine step size and direction.

## Specification

Implement a function in **`newton.py`** with the following signature:

```python
def optimize(f, x0, tol=1e-6, max_iter=100):
```

**Parameters:**
- `f` — a callable representing the objective function $f(x)$
- `x0` — initial guess (float)
- `tol` — convergence tolerance (default `1e-6`)
- `max_iter` — maximum number of iterations (default `100`)

**Returns** a dict containing at least:
- `'x'` — the estimated optimum
- `'converged'` — `True` if the method converged, `False` otherwise

**Requirements:**
- Compute derivatives using **finite difference approximation** (no autodiff, no symbolic math).
- **Stopping criterion:** $|x_t - x_{t-1}| < \text{tol}$

## Suggested Tasks

1. Implement `newton.py` following the spec above.
2. Write tests in `test_newton.py` and run them with `pytest`.
3. Extend to multivariate Newton's method (using the Hessian).

---

*This exercise is adapted from the [UC Berkeley Computational Skills Workshop](https://github.com/niclas-dern/computational-skills-workshop).*
