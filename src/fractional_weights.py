"""beta/phi weight generation for the Caputo memory sum. See plan.md 1.5, 1.9."""

import numpy as np


def betas(alpha: float, n_max: int) -> np.ndarray:
    """beta[j] = (j+1)**(1-alpha) - j**(1-alpha), j = 0..n_max"""
    raise NotImplementedError


def phis(beta: np.ndarray) -> np.ndarray:
    """phi[j] = beta[j] - beta[j+1], j = 0..len(beta)-2"""
    raise NotImplementedError
