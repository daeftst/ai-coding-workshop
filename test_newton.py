# ABOUTME: Tests for the Newton's method optimizer in newton.py.
# ABOUTME: Covers convergence, return structure, iteration limits, and tolerance.

import math
import pytest
import numpy as np
from newton import optimize, optimize_nd


def test_return_structure():
    result = optimize(lambda x: x ** 2, x0=1.0)
    assert 'x' in result
    assert 'converged' in result


def test_quadratic_converges_to_minimum():
    # f(x) = (x - 3)^2, minimum at x=3
    result = optimize(lambda x: (x - 3) ** 2, x0=0.0)
    assert result['converged'] is True
    assert abs(result['x'] - 3.0) < 1e-5


def test_quartic_converges_from_positive_start():
    # f(x) = x^4 - 4x^2, local minima at x=±sqrt(2)
    result = optimize(lambda x: x ** 4 - 4 * x ** 2, x0=1.5)
    assert result['converged'] is True
    assert abs(result['x'] - math.sqrt(2)) < 1e-5


def test_sin_converges_to_local_minimum():
    # sin(x) has a minimum near x = -pi/2 + 2k*pi; start near -pi/2
    result = optimize(math.sin, x0=-1.0)
    assert result['converged'] is True
    assert abs(result['x'] - (-math.pi / 2)) < 1e-5


def test_converged_flag_true_on_easy_case():
    result = optimize(lambda x: x ** 2, x0=1.0)
    assert result['converged'] is True


def test_converged_flag_false_when_max_iter_reached():
    # max_iter=1 is not enough to converge from x0=10
    result = optimize(lambda x: (x - 100) ** 2, x0=0.0, max_iter=1)
    assert result['converged'] is False


def test_tolerance_respected():
    tol = 1e-4
    result = optimize(lambda x: (x - 5) ** 2, x0=0.0, tol=tol)
    assert result['converged'] is True
    assert abs(result['x'] - 5.0) < tol


def test_near_zero_second_derivative_returns_early():
    # f(x) = x (linear), f''(x) = 0 everywhere
    result = optimize(lambda x: x, x0=1.0)
    assert result['converged'] is False


# --- Multivariate tests ---

def test_nd_return_structure():
    result = optimize_nd(lambda x: x[0] ** 2 + x[1] ** 2, x0=[1.0, 1.0])
    assert 'x' in result
    assert 'converged' in result


def test_nd_2d_quadratic_at_origin():
    # f(x, y) = x^2 + y^2, minimum at (0, 0)
    result = optimize_nd(lambda x: x[0] ** 2 + x[1] ** 2, x0=[2.0, -3.0])
    assert result['converged'] is True
    assert np.linalg.norm(result['x']) < 1e-5


def test_nd_2d_shifted_quadratic():
    # f(x, y) = (x - 2)^2 + (y - 3)^2, minimum at (2, 3)
    result = optimize_nd(lambda x: (x[0] - 2) ** 2 + (x[1] - 3) ** 2, x0=[0.0, 0.0])
    assert result['converged'] is True
    assert abs(result['x'][0] - 2.0) < 1e-5
    assert abs(result['x'][1] - 3.0) < 1e-5


def test_nd_3d_quadratic():
    # f(x, y, z) = x^2 + 2y^2 + 3z^2, minimum at (0, 0, 0)
    result = optimize_nd(lambda x: x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2, x0=[1.0, 1.0, 1.0])
    assert result['converged'] is True
    assert np.linalg.norm(result['x']) < 1e-5


def test_nd_converged_flag_false_when_max_iter_reached():
    # max_iter=1 won't converge from far away
    result = optimize_nd(lambda x: (x[0] - 100) ** 2 + (x[1] - 100) ** 2, x0=[0.0, 0.0], max_iter=1)
    assert result['converged'] is False


def test_nd_singular_hessian_returns_early():
    # f(x, y) = x^2 only — Hessian is singular (zero second derivative in y)
    result = optimize_nd(lambda x: x[0] ** 2, x0=[1.0, 1.0])
    assert result['converged'] is False
