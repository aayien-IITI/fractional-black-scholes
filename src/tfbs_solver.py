"""Full assembled time-fractional Black-Scholes Crank-Nicolson solver,
eqs. (3.9)-(3.14). See plan.md Section 1 for all resolved conventions."""

import numpy as np


def solve_tfbs(K, r, delta, sigma, T, Smax, L, N, alpha,
                memory_boundary_index: str = "n+1") -> np.ndarray:
    """Full tfBS Crank-Nicolson solver per eq (3.9)-(3.14).
    Returns U of shape (N+1, L+1): U[n, l] = V(S_l, t_n), t_n = time-to-maturity.
    U[N, :] is today's option-value curve."""
    raise NotImplementedError
