"""Reproduce Fig. 5 (maturity payoffs, delta = 0.045/0.085) and Figs. 6-7
(payoff surfaces). Resolves the delta-labeling discrepancy noted in plan.md
1.10 by producing both delta variants explicitly for both Fig. 6 and Fig. 7
(the paper capitions both surfaces "delta = 0.085", even though Example 5.2
specifies two dividend yields and Fig. 5 correctly shows both -- almost
certainly a duplicated/mislabeled figure in the original publication).
Parameters: K=200, r=0.065, sigma=0.025, T=1, Smax=600, L=100, N=100.
Owned by Person B. See plan.md Phase 8.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tfbs_solver import solve_tfbs

K, R, SIGMA, T, SMAX, L, N = 200.0, 0.065, 0.025, 1.0, 600.0, 100, 100
DELTAS = [0.045, 0.085]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")


def grid():
    h = SMAX / L
    return np.arange(0, L + 1) * h


def fig5_maturity_payoffs():
    S = grid()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, delta in zip(axes, DELTAS):
        for alpha in ALPHAS:
            U = solve_tfbs(K, R, delta, SIGMA, T, SMAX, L, N, alpha)
            ax.plot(S, U[N, :], label=f"alpha = {alpha}", marker="o", markevery=5, markersize=3)
        ax.set_title(f"European Put payoffs at delta = {delta}")
        ax.set_xlabel("S")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("V(S,t)")
    fig.suptitle("Fig. 5: Example 5.2 maturity payoffs (t = T)")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig5_example_5_2_maturity_payoffs.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def payoff_surface_figures():
    """Figs. 6-7. Unlike the paper (which captions both "delta=0.085"), we
    generate both delta variants for each figure slot -- see plan.md 1.10."""
    S = grid()
    t = np.arange(0, N + 1) * (T / N)
    Sg, Tg = np.meshgrid(S, t)

    for fig_idx, delta in zip((6, 7), DELTAS):
        fig = plt.figure(figsize=(16, 9))
        for i, alpha in enumerate(ALPHAS, start=1):
            U = solve_tfbs(K, R, delta, SIGMA, T, SMAX, L, N, alpha)
            ax = fig.add_subplot(2, 3, i, projection="3d")
            ax.plot_surface(Sg, Tg, U, cmap="viridis", linewidth=0, antialiased=True)
            ax.set_title(f"alpha = {alpha}")
            ax.set_xlabel("S")
            ax.set_ylabel("t")
            ax.set_zlabel("V(S,t)")
        fig.suptitle(f"Fig. {fig_idx}: Example 5.2 general payoffs (delta = {delta})")
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"fig{fig_idx}_example_5_2_surface_delta_{delta}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig5_maturity_payoffs()
    payoff_surface_figures()
