"""Matrix diagonal assembly for the tfBS Crank-Nicolson scheme, eqs. (3.10)-(3.12).
See plan.md 1.4, 1.7: B is constant across steps (n>=1, uses beta1), A must be
rebuilt every step (depends on t_next^(1-alpha))."""

import numpy as np


def build_B(l_grid, sigma, alpha, rho_alpha, beta1):
    """Returns (a, b, c) diagonals for matrix B (n>=1 case, eq 3.12). Constant across time steps."""
    raise NotImplementedError


def build_A(l_grid, sigma, alpha, r, q, t_next, rho_alpha, beta_self):
    """Returns (a, b, c) diagonals for matrix A at time level t_next.
    beta_self is beta0 for the n=0 step (eq 3.10), beta1 for n>=1 (eq 3.12)."""
    raise NotImplementedError
