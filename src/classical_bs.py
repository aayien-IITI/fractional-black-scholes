"""Analytic and plain Crank-Nicolson classical Black-Scholes, for the alpha=1
tiebreaker check against the fractional solver. See plan.md 1.9, Phase 1-2."""

import numpy as np
from scipy.stats import norm

from src.thomas_solver import thomas


def bs_put(S, K, r, delta, sigma, T):
    """Analytic European put (Black-Scholes formula, continuous dividend)."""
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore"):
        d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-delta * T) * norm.cdf(-d1)
    return np.where(S <= 0, K * np.exp(-r * T), price)


def cn_bs_put(K, r, delta, sigma, T, Smax, L, N) -> np.ndarray:
    """Plain (non-fractional) Crank-Nicolson finite-difference solver.
    Returns U[N+1, L+1]. Used to validate grid/boundary handling in isolation.

    Marches forward in time-to-maturity tau = T - t_calendar, tau_n = n*k,
    from the maturity payoff (n=0) to today's value (n=N), matching the
    tfBS solver's grid convention (plan.md 1.2-1.3). The PDE in tau is
        dV/dtau = 0.5*sigma^2*S^2*d2V/dS2 + q*S*dV/dS - r*V,   q = r - delta,
    discretized with standard Crank-Nicolson central differences.
    """
    q = r - delta
    h = Smax / L
    k = T / N
    l = np.arange(0, L + 1)

    A_l = 0.5 * sigma ** 2 * l ** 2 - 0.5 * q * l
    B_l = -(sigma ** 2) * l ** 2 - r
    C_l = 0.5 * sigma ** 2 * l ** 2 + 0.5 * q * l

    interior = slice(1, L)
    a = -(k / 2) * A_l[interior]
    b = 1 - (k / 2) * B_l[interior]
    c = -(k / 2) * C_l[interior]

    U = np.zeros((N + 1, L + 1))
    S = l * h
    U[0, :] = np.maximum(K - S, 0.0)

    t = np.arange(0, N + 1) * k
    U[:, 0] = K * np.exp(-r * t)
    U[:, L] = 0.0

    for n in range(N):
        rhs = (
            (k / 2) * A_l[interior] * U[n, 0:L - 1]
            + (1 + (k / 2) * B_l[interior]) * U[n, 1:L]
            + (k / 2) * C_l[interior] * U[n, 2:L + 1]
        )
        rhs[0] += (k / 2) * A_l[1] * U[n + 1, 0]
        rhs[-1] += (k / 2) * C_l[L - 1] * U[n + 1, L]

        U[n + 1, 1:L] = thomas(a.copy(), b.copy(), c.copy(), rhs)

    return U
