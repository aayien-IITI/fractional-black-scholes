"""Analytic and plain Crank-Nicolson classical Black-Scholes, for the alpha=1
tiebreaker check against the fractional solver. See plan.md 1.9, Phase 1-2."""

import numpy as np


def bs_put(S, K, r, delta, sigma, T):
    """Analytic European put (Black-Scholes formula, continuous dividend)."""
    raise NotImplementedError


def cn_bs_put(K, r, delta, sigma, T, Smax, L, N) -> np.ndarray:
    """Plain (non-fractional) Crank-Nicolson finite-difference solver.
    Returns U[N+1, L+1]. Used to validate grid/boundary handling in isolation."""
    raise NotImplementedError
