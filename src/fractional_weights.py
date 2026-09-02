"""beta/phi weight generation for the Caputo memory sum. See plan.md 1.5, 1.9."""

import numpy as np


def betas(alpha: float, n_max: int) -> np.ndarray:
    """beta[j] = (j+1)**(1-alpha) - j**(1-alpha), j = 0..n_max.

    j=0 is special-cased to 0**(1-alpha) := 0 (its natural value for alpha < 1,
    extended by continuity to alpha = 1) rather than relying on Python/NumPy's
    0**0 = 1 convention, which would wrongly give beta[0] = 0 at alpha = 1
    instead of the required beta[0] = 1 (see plan.md 1.9 / paper Remark 3.1).
    """
    j = np.arange(0, n_max + 1, dtype=float)
    j_pow = np.where(j == 0, 0.0, j ** (1 - alpha))
    return (j + 1) ** (1 - alpha) - j_pow


def phis(beta: np.ndarray) -> np.ndarray:
    """phi[j] = beta[j] - beta[j+1], j = 0..len(beta)-2"""
    return beta[:-1] - beta[1:]
