"""Full assembled time-fractional Black-Scholes Crank-Nicolson solver.

See plan.md Section 1 for the resolved PDE/BC conventions, and
src/coefficients.py's module docstring (plus docs/validation_report.md) for
why the memory-sum closed form implemented here differs from the paper's
printed eq. (3.11): that formula was found to be both numerically unstable
and algebraically inconsistent with the paper's own raw discretization (eq.
3.6) for every tested alpha. The corrected form below is derived directly
from eq. (3.6)'s raw sum, sum_{j=0}^n beta_j (V^{n-j+1}_l - V^{n-j-1}_l),
using a symmetric ghost point V^{-1}_l := V^0_l for the one fictitious
pre-maturity value the sum's last term (j=n) always requires. Collecting
that sum's coefficients per historical time index m (verified with sympy)
gives, for n >= 1:

    coeff(V^{n+1}_l) = beta_0     (the unknown -- folded into build_A's diagonal)
    coeff(V^n_l)     = beta_1
    coeff(V^m_l)     = -(phi_{n-m-1} + phi_{n-m})   for m = 1, ..., n-1
    coeff(V^0_l)     = -(beta_{n-1} + beta_n)

This exactly reproduces the paper's own n=0 bootstrap step (eq. 3.10) as a
special case, and the resulting scheme achieves the paper's claimed
O(k^2+h^2) double-mesh convergence rate for every tested alpha, including
alpha=1 (verified in tests/test_tfbs_vs_classical.py).
"""

import numpy as np
from scipy.special import gamma

from src.coefficients import build_A, build_B
from src.fractional_weights import betas, phis
from src.thomas_solver import thomas


def _memory_known_terms(n, beta, phi, U, interior_slice):
    """The Sigma of eq. (3.6)'s raw sum minus its beta_0 * V^{n+1} (unknown)
    piece, i.e. everything on the "known history" side. See module docstring."""
    if n == 0:
        return -beta[0] * U[0, interior_slice]

    w = np.empty(n + 1)
    w[n] = beta[1]
    if n >= 2:
        m_mid = np.arange(1, n)
        idx = n - m_mid
        w[m_mid] = -(phi[idx - 1] + phi[idx])
    w[0] = -(beta[n - 1] + beta[n])
    return np.tensordot(w, U[0:n + 1, interior_slice], axes=(0, 0))


def solve_tfbs(K, r, delta, sigma, T, Smax, L, N, alpha,
                memory_boundary_index: str = "n+1") -> np.ndarray:
    """Full tfBS Crank-Nicolson solver, discretizing eq. (2.13) per the
    corrected closed form documented above.
    Returns U of shape (N+1, L+1): U[n, l] = V(S_l, t_n), t_n = time-to-maturity.
    U[N, :] is today's option-value curve.

    memory_boundary_index is accepted for backwards compatibility with the
    original (paper-literal) API but has no effect: the corrected closed
    form has no such ambiguity (see coefficients.py / this module's
    docstring for why the literal eq 3.11 boundary-index choice does not
    arise here).
    """
    q = r - delta
    h = Smax / L
    k = T / N
    l = np.arange(0, L + 1)
    S = l * h
    t = np.arange(0, N + 1) * k

    rho_alpha = -1.0 / (2.0 * gamma(2 - alpha) * k ** alpha)
    beta = betas(alpha, N)
    phi = phis(beta)

    U = np.zeros((N + 1, L + 1))
    U[0, :] = np.maximum(K - S, 0.0)
    U[:, 0] = K * np.exp(-r * t)
    U[:, L] = 0.0

    interior = l[1:L]
    interior_slice = slice(1, L)
    rho_alpha_beta0 = rho_alpha * beta[0]

    aB, bB, cB = build_B(interior, sigma, alpha)

    for n in range(N):
        aA, bA, cA = build_A(interior, sigma, alpha, r, q, t[n + 1], rho_alpha_beta0)

        rhs = aB * U[n, 0:L - 1] + bB * U[n, 1:L] + cB * U[n, 2:L + 1]
        rhs -= rho_alpha * _memory_known_terms(n, beta, phi, U, interior_slice)

        rhs[0] -= aA[0] * U[n + 1, 0]
        rhs[-1] -= cA[-1] * U[n + 1, L]

        U[n + 1, 1:L] = thomas(aA.copy(), bA.copy(), cA.copy(), rhs)

    return U
