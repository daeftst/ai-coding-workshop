# ABOUTME: Tests for the Newton's method optimizer in newton.py.
# ABOUTME: Covers convergence, return structure, iteration limits, and tolerance.

import math
import pytest
from newton import optimize


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
