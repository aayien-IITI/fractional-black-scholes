"""Matrix diagonal assembly for the tfBS Crank-Nicolson scheme.

See plan.md 1.4-1.7 and the validation report (docs/validation_report.md,
section "Coefficient closed form") for why this module does NOT literally
reproduce eqs. (3.10)-(3.12) as printed. In short: the printed n>=1
coefficients (a_{n+1}, b_{n+1}, c_{n+1}, a_n, b_n, c_n) use beta_1 in the
diagonal (b) terms. That formula was verified (both by direct numerical
test and by symbolic substitution against the raw discretization eq. 3.6)
to be numerically unstable and algebraically inconsistent with the paper's
own raw sum -- it blows up for every tested alpha, not just alpha=1. This
module instead assembles the coefficients from a corrected closed form,
derived directly from the raw fractional-derivative sum (eq. 3.6) using a
symmetric ghost-point convention for the one fictitious pre-maturity value
(V^{-1}_l := V^0_l) that the raw sum's last term always requires. This
correction was verified symbolically (sympy) and reproduces the paper's own
n=0 bootstrap step (eq. 3.10) exactly, and achieves the paper's claimed
O(k^2+h^2) double-mesh convergence rate for all tested alpha, including
alpha=1.
"""

import numpy as np
from scipy.special import gamma


def build_B(l_grid, sigma, alpha):
    """Returns (a, b, c) diagonals for the "old level" (diffusion-only)
    contribution, used for both the n=0 step (eq 3.10's a0/b0/c0) and the
    n>=1 step (eq 3.12's a_n/b_n/c_n) -- the two coincide under the
    corrected closed form, so a single constant B matrix serves every step.

    Unlike the paper's printed formula, this diffusion-only piece carries
    NO explicit beta/rho_alpha term -- that contribution is folded entirely
    into build_A's diagonal (via beta_0, which is always 1) plus the
    explicit historical memory sum computed by the caller (tfbs_solver).
    """
    l = np.asarray(l_grid, dtype=float)
    g1a = gamma(1 + alpha)
    a = -sigma ** 2 * l ** 2 * g1a / 4
    b = sigma ** 2 * l ** 2 * g1a / 2
    c = -sigma ** 2 * l ** 2 * g1a / 4
    return a, b, c


def build_A(l_grid, sigma, alpha, r, q, t_next, rho_alpha_beta0):
    """Returns (a, b, c) diagonals for the new-level ("A") matrix at time
    level t_next. rho_alpha_beta0 is rho_alpha * beta_0 -- the diagonal's
    fractional-derivative contribution. The corrected closed form always
    uses beta_0 (=1 identically, by construction -- see fractional_weights)
    here, never beta_1: it is the coefficient of the unknown V^{n+1} in the
    raw sum (eq 3.6) for every n, not just n=0 -- see the validation report."""
    l = np.asarray(l_grid, dtype=float)
    g1a = gamma(1 + alpha)
    g2a = gamma(2 - alpha)
    tp = t_next ** (1 - alpha) / g2a

    a = sigma ** 2 * l ** 2 * g1a / 4 - q * l * tp
    b = rho_alpha_beta0 - r * tp - sigma ** 2 * l ** 2 * g1a / 2
    c = sigma ** 2 * l ** 2 * g1a / 4 + q * l * tp
    return a, b, c
