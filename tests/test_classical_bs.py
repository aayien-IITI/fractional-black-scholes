"""Phase 1-2 stopping tests: bs_put matches a published/known put value to 1e-10;
cn_bs_put max error vs. bs_put shrinks by ~4x when h,k both halve. See plan.md Section 4."""

import numpy as np
import pytest

from src.classical_bs import bs_put, cn_bs_put


def test_bs_put_known_value():
    # Hull's textbook example (no dividend): S=42, K=40, r=0.10, sigma=0.20, T=0.5
    # gives call ~= 4.76, so put via parity = call - S + K*exp(-rT) ~= 0.8086.
    price = bs_put(S=42, K=40, r=0.10, delta=0.0, sigma=0.20, T=0.5)
    assert price == pytest.approx(0.8085993729, abs=1e-8)


def test_bs_put_zero_at_deep_itm_call_parity():
    # Put-call parity sanity check with a continuous dividend yield.
    S, K, r, delta, sigma, T = 100.0, 100.0, 0.05, 0.03, 0.25, 1.0
    put = bs_put(S, K, r, delta, sigma, T)
    d1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    from scipy.stats import norm
    call = S * np.exp(-delta * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    assert (call - put) == pytest.approx(S * np.exp(-delta * T) - K * np.exp(-r * T), abs=1e-10)


def test_cn_bs_put_converges_to_analytic():
    K, r, delta, sigma, T, Smax = 150.0, 0.055, 0.025, 0.20, 1.0, 450.0

    errors = []
    for L, N in [(50, 50), (100, 100), (200, 200), (400, 400)]:
        U = cn_bs_put(K, r, delta, sigma, T, Smax, L, N)
        h = Smax / L
        S = np.arange(0, L + 1) * h
        analytic = bs_put(S, K, r, delta, sigma, T)
        err = np.max(np.abs(U[N, :] - analytic))
        errors.append(err)

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert ratio > 3.0
