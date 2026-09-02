"""Phase 3 stopping test: beta[0]=1, strictly decreasing, positive;
at alpha=1, beta[0]=1 and all later beta[j]=0. See plan.md Section 4."""

import numpy as np
import pytest

from src.fractional_weights import betas, phis


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_beta_properties(alpha):
    beta = betas(alpha, 50)
    assert beta[0] == pytest.approx(1.0)
    assert np.all(beta > 0)
    assert np.all(np.diff(beta) < 0)
    assert beta[-1] < beta[1]


def test_beta_decays_to_zero():
    beta = betas(0.5, 5000)
    assert beta[-1] < beta[100] < beta[10]


def test_beta_alpha_one_degenerates():
    beta = betas(1.0, 20)
    assert beta[0] == pytest.approx(1.0)
    assert np.allclose(beta[1:], 0.0)


def test_phi_properties():
    alpha = 0.5
    n = 30
    beta = betas(alpha, n)
    phi = phis(beta)
    assert phi[0] == pytest.approx(1 - beta[1])
    cumsum = np.cumsum(phi)
    expected = 1 - beta[1:n + 1]
    assert np.allclose(cumsum, expected)
