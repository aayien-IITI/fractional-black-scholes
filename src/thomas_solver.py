"""Thomas algorithm for tridiagonal systems. See plan.md Section 3."""

import numpy as np


def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system. a[0] and c[-1] unused (sub/super-diagonals)."""
    raise NotImplementedError
