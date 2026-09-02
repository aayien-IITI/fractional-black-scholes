"""Double-mesh convergence study reproducing Tables 1-4. Refines h and k
together (L and N doubling in lockstep), not N alone -- see plan.md Section 6
for why: the scheme's error is O(k^2+h^2), so freezing L=100 while refining N
lets the spatial error floor dominate and the measured rate collapse well
before reaching the fine end of the N range.

Also runs the cheap stability sanity check from plan.md Section 6: a fixed
fine spatial grid with deliberately large time steps (small N) should stay
bounded and non-oscillatory, which is the observable content of "the scheme
is unconditionally stable" (Theorem 4.1) -- this is a sanity check, not a
proof.

Owned by Person B. See plan.md Phase 9.
"""

import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tfbs_solver import solve_tfbs

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "tables")

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_VALUES = [100, 200, 400, 800, 1600]

EXAMPLE_5_1 = dict(K=150.0, r=0.055, delta=0.025, sigma=0.01, T=1.0, Smax=450.0)
EXAMPLE_5_2 = dict(K=200.0, r=0.065, delta=0.085, sigma=0.025, T=1.0, Smax=600.0)


def double_mesh_error(params, alpha, N):
    """E_N = max_l |V_l^N(fine grid restricted to coarse nodes) - V_l^N(coarse grid)|"""
    L = N
    U_coarse = solve_tfbs(**params, L=L, N=N, alpha=alpha)
    U_fine = solve_tfbs(**params, L=2 * L, N=2 * N, alpha=alpha)
    return np.max(np.abs(U_coarse[N, :] - U_fine[2 * N, ::2]))


def build_tables(params, label, n_values=N_VALUES):
    """Returns (errors, rates) dicts keyed by alpha, matching Tables 1/3 and
    2/4's shape: errors[alpha] = [E_100, E_200, ...], rates[alpha] = [p_200, p_400, ...]"""
    errors = {}
    rates = {}
    for alpha in ALPHAS:
        t0 = time.time()
        errs = [double_mesh_error(params, alpha, N) for N in n_values]
        errors[alpha] = errs
        rates[alpha] = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        print(f"[{label}] alpha={alpha}: errors={errs} rates={rates[alpha]} "
              f"({time.time()-t0:.1f}s)")
    return errors, rates


def write_csv(path, errors, rates, n_values=N_VALUES):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha"] + [f"N={N}" for N in n_values])
        for alpha, errs in errors.items():
            writer.writerow([alpha] + [f"{e:.4e}" for e in errs])
        writer.writerow([])
        writer.writerow(["alpha"] + [f"rate(N={n_values[i]}->{n_values[i+1]})"
                                      for i in range(len(n_values) - 1)])
        for alpha, rs in rates.items():
            writer.writerow([alpha] + [f"{r:.2f}" for r in rs])
    print("wrote", path)


def stability_check(params, label, n_values=(10, 20, 40)):
    """Section 6's cheap stability sanity check: large time steps on a fixed
    fine spatial grid should stay bounded and non-oscillatory."""
    print(f"\n[{label}] stability check (large time steps, L=200 fixed):")
    for alpha in (0.1, 0.5, 0.9, 1.0):
        for N in n_values:
            U = solve_tfbs(**params, L=200, N=N, alpha=alpha)
            today = U[N, :]
            bounded = np.all(np.isfinite(today)) and today.max() <= params["K"] * 1.01
            oscillation = np.max(np.abs(np.diff(today, 2)))
            print(f"  alpha={alpha} N={N}: bounded={bounded} max={today.max():.2f} "
                  f"2nd-diff-max={oscillation:.4f}")


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)

    errors_51, rates_51 = build_tables(EXAMPLE_5_1, "Example 5.1")
    write_csv(os.path.join(TABLES_DIR, "table1_table2_example_5_1.csv"), errors_51, rates_51)

    errors_52, rates_52 = build_tables(EXAMPLE_5_2, "Example 5.2")
    write_csv(os.path.join(TABLES_DIR, "table3_table4_example_5_2.csv"), errors_52, rates_52)

    stability_check(EXAMPLE_5_1, "Example 5.1")
    stability_check(EXAMPLE_5_2, "Example 5.2")
