# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A workshop exercise: implement univariate Newton's method for optimization in `newton.py`, write tests in `test_newton.py`, and optionally create a Jupyter notebook for visualization.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
pytest                     # run all tests
pytest test_newton.py      # run a specific test file
pytest test_newton.py::test_name  # run a single test
jupyter notebook           # open Jupyter for visualization work
```

## Implementation Spec

The target function in `newton.py`:

```python
def optimize(f, x0, tol=1e-6, max_iter=100):
```

- `f` — callable univariate objective function
- `x0` — initial guess (float)
- Returns a dict with at least `'x'` (estimated optimum) and `'converged'` (bool)
- Derivatives must use **finite difference approximation** (no autodiff, no symbolic math)
- Stopping criterion: `|x_t - x_{t-1}| < tol`
- Update rule: `x_{t+1} = x_t - f'(x_t) / f''(x_t)`
