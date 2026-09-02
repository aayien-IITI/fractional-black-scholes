"""Phase 4 stopping test: thomas() matches scipy.linalg.solve_banded to
machine precision on random diagonally-dominant systems. See plan.md Section 4."""

import numpy as np
from scipy.linalg import solve_banded

from src.thomas_solver import thomas


def _random_diag_dominant_system(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1, 1, n)
    c = rng.uniform(-1, 1, n)
    b = (np.abs(a) + np.abs(c) + rng.uniform(1, 2, n))
    d = rng.uniform(-5, 5, n)
    a[0] = 0.0
    c[-1] = 0.0
    return a, b, c, d


def test_thomas_matches_solve_banded():
    for seed in range(10):
        n = 50
        a, b, c, d = _random_diag_dominant_system(n, seed)

        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]
        ab[1, :] = b
        ab[2, :-1] = a[1:]

        expected = solve_banded((1, 1), ab, d)
        actual = thomas(a, b, c, d)

        assert np.allclose(actual, expected, atol=1e-10, rtol=1e-10)
