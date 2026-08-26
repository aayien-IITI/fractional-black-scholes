# Reproduction Plan: Time-Fractional Black–Scholes PDE

**Paper:** S.M. Nuugulu, F. Gideon, K.C. Patidar, *"A robust numerical scheme
for a time-fractional Black-Scholes partial differential equation describing
stock exchange dynamics"*, Chaos, Solitons and Fractals 145 (2021) 110753.

This plan was written from a direct read of the paper PDF itself (equations,
figures, tables, and discussion all checked against the primary source) — not
from any intermediate summary. Section 1 below records every place the paper
is internally ambiguous or where a literal transcription would be wrong, and
gives the resolution to implement. **Both people must read Section 1 before
writing any code** — it is the source of truth; do not re-derive from the
equations independently, since that risks two different (and differently
wrong) implementations.

---

## 0. What the paper actually does (for orientation)

Classical Black–Scholes assumes the stock price is memoryless (a standard
Brownian motion). This paper replaces the time derivative in the option
pricing PDE with a **Caputo fractional derivative of order α ∈ (0, 1]**,
which lets the model encode "memory" in the price process. At α = 1 the
model collapses back to ordinary Black–Scholes.

The paper:
1. Derives the time-fractional Black–Scholes (tfBS) PDE, eq. (2.13).
2. Builds a Crank–Nicolson-style finite-difference scheme to solve it, eqs.
   (3.9)–(3.14), by discretizing the fractional time derivative with a
   weighted sum over all previous time levels (the "memory" term) and the
   spatial derivatives with standard central differences.
3. Proves the scheme is unconditionally stable (Section 4.1) and converges
   at rate O(k² + h²) (Section 4.2).
4. Demonstrates it on two European put option examples (Section 5, Tables
   1–4, Figs. 1–7).

Our job is to implement the scheme and reproduce Tables 1–4 and Figs. 1–7.

---

## 1. Resolved conventions and known paper issues

### 1.1 The PDE, initial condition, and boundary conditions (eqs. 2.13–2.14)

```
∂^α V/∂t^α = (r·V − q·S·∂V/∂S) · t^(1-α)/Γ(2-α) − ((1+α)/2)·σ²S²·∂²V/∂S²
q = r − δ,   0 < α ≤ 1
```
```
V(S, 0)      = max(K − S, 0)      (payoff)
V(0, t)      = K·e^(−r(T−t))
lim_{S→∞} V(S, t) = 0
```

### 1.2 Time grid is time-to-maturity, not calendar time

The algorithm marches `n = 0 → N` starting from `V⁰ = payoff` — i.e. `n=0`
is maturity and `n=N` is "now" (today's price, the number we actually
report). This only makes sense if the grid variable `tₙ = nk` is
**time-to-maturity** `τ = T − t_calendar`, not calendar time. This is the
standard τ = T−t substitution used throughout the numerical option-pricing
literature to turn a backward terminal-value problem into a forward
initial-value one; the paper applies it implicitly without renaming the
symbol, which causes the boundary-condition inconsistency below.

### 1.3 Left boundary condition — implement this, not the literal eq. (2.14) formula

Eq. (2.14) prints `V(0,t) = K·e^{-r(T-t)}` in terms of the *original*
calendar time. If you plug the grid variable `tₙ = nk` (which is really
time-to-maturity, per 1.2) directly into that formula, you get `K·e^{-rT}`
at `n=0` — wrong, since a put with `S=0` at maturity is worth exactly `K`,
undiscounted. The version consistent with the time-to-maturity grid is:

```
V₀ⁿ = K · exp(−r · tₙ),   tₙ = n·k = time-to-maturity
```

This gives `K` at `n=0` (correct) and `K·e^{-rT}` at `n=N` (correct: today's
discounted value). Verify this against the α=1 check in 1.6 before trusting
it on the full examples.

### 1.4 Use `Γ(2-α)` (the Gamma function) everywhere

Confirmed directly from the PDF: eq. (2.13) and every coefficient formula in
(3.1)–(3.14) show `Γ(2-α)` (and `Γ(1+α)` in the diffusion coefficient)
explicitly, not bare arithmetic. Use `scipy.special.gamma`.

### 1.5 Fractional memory term (eq. 3.6 / 3.9)

```
∂^α V(Sₗ,tₙ₊₁)/∂t^α ≈ ρα · Σ_{j=0}^{n} βⱼ (V^{n-j+1}_l − V^{n-j-1}_l)
ρα = −1 / (2·Γ(2-α)·k^α)
βⱼ = (j+1)^{1-α} − j^{1-α},   1 = β₀ > β₁ > β₂ > ... → 0
```

This is derived (not assumed) in the paper via a centered-difference
discretization of the Caputo integral that would naively need a
nonexistent `V⁻¹` term; the paper telescopes that away (their eq. 3.1→3.2
index shift) before handing you eqs. (3.9)–(3.14). **Implement the closed
form (3.9)–(3.14) directly — do not re-derive the raw centered sum.**

### 1.6 Genuine unresolved ambiguity in the paper — the boundary/memory constant `Cⁿ`

This is a real inconsistency in the published paper (confirmed by direct
reading, both instances present verbatim), not a transcription error:

- Eq. (3.11), inline, for `n ≥ 1`: last term is `ρα · β_{n+1} · V⁰_l`.
- The formal definition given right after eq. (3.14): `Cⁿ = ρα · βₙ · U⁰_l`
  (no `+1`).

Implement it as a single named constant (`memory_boundary_index`, either
`"n"` or `"n+1"`) so it can be flipped with one line while debugging against
the α=1 check (1.9 below settles which one is right).

### 1.7 Matrix assembly: `B` is constant, `A` must be rebuilt every step

Direct read of eqs. (3.10)–(3.12) confirms:
- For `n ≥ 1`, the `A`-side coefficients (`aₙ₊₁, bₙ₊₁, cₙ₊₁`) depend on
  `t_{n+1}^{1-α}` (through the `r` and `q` terms) **and** always use `β₁`
  (fixed, not `β_{n+1}`) in `bₙ₊₁`. So `A` changes every step only because
  of the `t_{n+1}^{1-α}` factor.
- The `B`-side coefficients (`aₙ, bₙ, cₙ`) depend only on `σ², l², (1+α)`
  diffusion terms and `ρα·β₁` — no `t`, `r`, or `q`. **`B` is the same
  matrix at every step ≥1** and should be assembled once outside the time
  loop. (The `n=0` step, eq. 3.10, has its own separate `a₀,b₀,c₀` using
  `β₀` instead of `β₁` — a one-off, not part of the constant `B` used for
  `n≥1`.)
- The scheme is not uniformly Crank–Nicolson: the reaction (`rV`) and
  convection (`qS·∂V/∂S`) terms are fully implicit (new time layer only,
  visible in eq. 3.9's RHS `t^{1-α}/Γ(2-α)` block); only the diffusion term
  is CN-averaged between old and new layers (eq. 3.8). Use the coefficient
  formulas exactly as printed in eqs. (3.10)–(3.12); do not re-derive signs.

### 1.8 Grid does not place the strike on a node — expected, not a bug

Example 5.1: `h = 450/100 = 4.5`, so `K/h = 150/4.5 = 33.33` (off-node).
Example 5.2: `h = 600/100 = 6`, so `K/h = 200/6 = 33.33` (off-node).
Both are exactly as specified in the paper (§5) and the paper still reports
convergence rates approaching 2 in Tables 1–4. Replicate the paper's grid
as-is for the required tables; do not "fix" this. An optional side-study
with `L` chosen so `K` lands on a node (e.g. `L=90` for Example 5.1 gives
`h=5`, `K/h=30`) can quantify the effect separately (§6 below).

### 1.9 The α = 1 check is the tiebreaker for every remaining ambiguity

At `α=1`: `β₀=1` and every later `βⱼ=0` (from the `βⱼ` formula and Remark
3.1 in the paper), so the memory sum vanishes and the scheme should
collapse to an ordinary Crank–Nicolson Black–Scholes solver. Build this
comparison early (Phase 1 + Phase 6 below) and use it to:
- decide 1.6's `Cⁿ` index ambiguity,
- catch a flipped diffusion sign (shows up as blow-up within a few steps),
- catch a boundary-convention error from 1.3 (shows up as an error
  concentrated near `S=0` that does not shrink under grid refinement).

### 1.10 Fig. 6 / Fig. 7 caption duplication in the paper

Both Fig. 6 and Fig. 7 (Example 5.2 general payoff surfaces) are captioned
"δ = 0.085" in the printed paper, even though Example 5.2 specifies two
dividend yields (δ = 0.045 and 0.085) and Fig. 5 (the corresponding maturity
payoffs) correctly shows both. This looks like a duplicated/mislabeled
figure in the original publication. When reproducing Figs. 6–7, generate
both δ = 0.045 and δ = 0.085 surfaces (matching the pattern of Figs. 2–4 for
Example 5.1, which do show three distinct δ values) and note the discrepancy
in the validation report rather than blindly reproducing a mislabeled pair.

---

## 2. Repository layout

```
fractional-black-scholes/
|-- plan.md                     (this file)
|-- README.md
|-- requirements.txt
|
|-- src/
|   |-- fractional_weights.py   # betas(alpha, N), phis(betas)
|   |-- thomas_solver.py        # thomas(a, b, c, d) -> x
|   |-- coefficients.py         # build_A(t_next, l, params), build_B(l, params) [constant]
|   |-- tfbs_solver.py          # solve_tfbs(params, alpha, L, N) -> U[N+1, L+1]
|   `-- classical_bs.py         # bs_put(...) analytic; cn_bs_put(...) plain CN solver
|
|-- tests/
|   |-- test_weights.py
|   |-- test_thomas.py
|   |-- test_classical_bs.py
|   `-- test_tfbs_vs_classical.py   # the alpha=1 tiebreaker check
|
|-- experiments/
|   |-- example_5_1.py
|   |-- example_5_2.py
|   `-- convergence.py
|
|-- results/
|   |-- tables/
|   `-- figures/
|
`-- notebooks/
    `-- reproduction.ipynb      # final narrative notebook, built last
```

Section 3's function contracts are frozen before either person starts
coding, so both sides can work in parallel against stable interfaces.

---

## 3. Shared module contracts (freeze together, first)

```python
# fractional_weights.py
def betas(alpha: float, n_max: int) -> np.ndarray:
    """beta[j] = (j+1)**(1-alpha) - j**(1-alpha), j = 0..n_max"""

def phis(beta: np.ndarray) -> np.ndarray:
    """phi[j] = beta[j] - beta[j+1], j = 0..len(beta)-2"""

# thomas_solver.py
def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve tridiagonal system. a[0] and c[-1] unused (sub/super-diagonals)."""

# classical_bs.py
def bs_put(S, K, r, delta, sigma, T) -> float | np.ndarray:
    """Analytic European put (Black-Scholes formula, continuous dividend)."""

def cn_bs_put(K, r, delta, sigma, T, Smax, L, N) -> np.ndarray:
    """Plain (non-fractional) Crank-Nicolson finite-difference solver.
    Returns U[N+1, L+1]. Used to validate grid/boundary handling in isolation."""

# coefficients.py
def build_B(l_grid, sigma, alpha, rho_alpha, beta1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (a, b, c) diagonals for matrix B (n>=1 case, eq 3.12). Constant across time steps."""

def build_A(l_grid, sigma, alpha, r, q, t_next, rho_alpha, beta_self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (a, b, c) diagonals for matrix A at time level t_next.
    beta_self is beta0 for the n=0 step (eq 3.10), beta1 for n>=1 (eq 3.12)."""

# tfbs_solver.py
def solve_tfbs(K, r, delta, sigma, T, Smax, L, N, alpha,
                memory_boundary_index: str = "n+1") -> np.ndarray:
    """Full tfBS Crank-Nicolson solver per eq (3.9)-(3.14).
    Returns U of shape (N+1, L+1): U[n, l] = V(S_l, t_n), t_n = time-to-maturity.
    U[N, :] is today's option-value curve."""
```

---

## 4. Build order with stopping tests (do not skip ahead)

| Phase | What | Stopping test |
|---|---|---|
| 1 | `classical_bs.bs_put` | Matches a published/known put value to 1e-10 |
| 2 | `classical_bs.cn_bs_put` | Max error vs. `bs_put` shrinks by ~4x when `h,k` both halve |
| 3 | `fractional_weights` | `β₀=1`, strictly decreasing, positive; at `α=1`, `β₀=1` and all later `βⱼ=0` |
| 4 | `thomas_solver.thomas` | Matches `scipy.linalg.solve_banded` to machine precision on random diagonally-dominant systems |
| 5 | `coefficients` + `tfbs_solver` assembled | Runs without error on a tiny grid (`L=10, N=10`) |
| 6 | **α=1 tiebreaker** | `solve_tfbs(..., alpha=1)` matches `cn_bs_put(...)` to near machine precision. If not: fix before touching Examples 5.1/5.2. Use this run to settle the 1.6 index ambiguity — try both, keep whichever matches. |
| 7 | Example 5.1 | Reproduces Fig. 1 (maturity payoffs, three δ) and Figs. 2–4 (payoff surfaces per α); smoothness split at α=1/2 noted in §5 discussion should be visible |
| 8 | Example 5.2 | Reproduces Fig. 5 (maturity payoffs, two δ) and Figs. 6–7 (payoff surfaces) — see 1.10 for the δ discrepancy to resolve here |
| 9 | Convergence study | Tables 1–4: max errors and rates approaching 2 as N grows, for α ∈ {0.1, ..., 1.0} (see Section 6 for correct refinement method) |
| 10 | Final notebook + report | All figures/tables reproduced, discussion written, ambiguity resolutions documented |

---

## 5. Division of work (two people)

**Person A — Numerical Core** (Phases 1–6, `src/`, `tests/`)
- `classical_bs.py`: analytic put formula + plain CN Black-Scholes solver
- `fractional_weights.py`: β and φ weight generation, with the α=1 degeneracy check
- `thomas_solver.py`: Thomas algorithm, tested against `scipy.linalg.solve_banded`
- `coefficients.py`: `build_A` / `build_B` exactly per eqs. (3.10)–(3.12), respecting
  1.4 (Γ function) and 1.7 (B constant for n≥1, A rebuilt per step)
- `tfbs_solver.py`: full assembled solver per eqs. (3.9)–(3.14), respecting 1.2,
  1.3, 1.6 (including resolving the `memory_boundary_index` ambiguity via the
  α=1 test)
- Owns Phase 6 (the α=1 tiebreaker) and must sign off on it before Person B's
  Example 5.1/5.2 numbers are treated as trustworthy

**Person B — Experiments, Validation, Figures** (Phases 7–10, `experiments/`, `results/`)
- `experiments/example_5_1.py`: reproduce Fig. 1 (three δ: 0.025/0.055/0.065) and
  Figs. 2–4 (payoff surfaces for α = 0.1/0.3/0.5/0.7/0.9, one figure per δ)
- `experiments/example_5_2.py`: reproduce Fig. 5 (two δ: 0.045/0.085) and Figs.
  6–7 (payoff surfaces); resolve the δ-labeling discrepancy noted in 1.10 by
  producing both δ variants explicitly
- `experiments/convergence.py`: implement the double-mesh convergence check
  (Section 6 below — refine `h` and `k` together, not `N` alone) and reproduce
  Tables 1–4 for both Example parameter sets across α ∈ {0.1, ..., 1.0}
- Validation checks: initial payoff exactly matches intrinsic value; boundary
  conditions hold per 1.3; β properties per Phase 3; grid convergence;
  robustness across all α in {0.1, ..., 1.0} (matches the paper's claim in §5
  that convergence is robust regardless of α)
- Optional robustness side-study from 1.8 (K on-node vs off-node grid)
- Final `notebooks/reproduction.ipynb` narrative and the validation report
  (paper's tables/figures vs. ours)

**Joint work before splitting:**
- Freeze Section 3's function signatures together — this is the interface
  contract that lets both people work in parallel without merge conflicts
- Both must independently read Section 1 and confirm agreement before coding
  starts, since the paper's ambiguities (1.6, 1.10) affect both halves

**Integration checkpoint:** Person A hands off a passing Phase 6 (α=1 match)
before Person B starts Phase 7. Everything after that point is independently
parallelizable (Example 5.1, Example 5.2, and the convergence study touch
disjoint files).

---

## 6. Convergence study — do this correctly (owned by Person B)

There is no closed-form solution for `α<1`, so use the double-mesh
principle: run at `N` and at `2N`, compare on the coarse grid's nodes.

**Correction to naive N-only refinement:** refine `h` and `k` together
(both `L` and `N` doubling in lockstep), not `N` alone while holding
`L=100` fixed. The scheme's error is `O(k²+h²)` per Theorem 4.4; freezing
`h` means the error eventually stalls at the spatial floor and the measured
rate collapses toward 0 at the fine end, even though the scheme is
genuinely second order. Use, e.g., `(L,N) = (100,100), (200,200), (400,400),
(800,800), (1600,1600)` — matching the paper's `N` values in Tables 1–4
while also doubling `L` proportionally — or at minimum explicitly justify
why `L=100` alone is already below the spatial error floor for these
parameters before freezing it.

```
E_N = max_l | V_l^N(fine grid restricted to coarse nodes) - V_l^N(coarse grid) |
p_N = log2( E_N / E_2N )
```

Target rates: Tables 1–2 (Example 5.1) show ~1.91→1.99 as N goes 100→1600
across all α; Tables 3–4 (Example 5.2) show ~1.95→2.01. Match this shape,
not exact digits.

For the stability claim (Theorem 4.1, §4.1), don't attempt the
Fourier/Parseval proof numerically. Run a fixed fine spatial grid with
deliberately large time steps (small `N`, e.g. 10/20/40) and confirm the
solution stays bounded and non-oscillatory — that's the observable content
of "unconditionally stable," and it's a cheap sanity check, not a formal
proof.

---

## 7. Final deliverables

1. `src/` — clean, tested Python implementation of the tfBS CN solver
2. `tests/` — passing tests for all six stopping-test phases above
3. `results/figures/` — Figs. 1–7 reproduced for both examples
4. `results/tables/` — Tables 1–4 (max errors + convergence rates) reproduced
5. `notebooks/reproduction.ipynb` — narrative walkthrough: equations →
   discretization → implementation → experiments → results
6. Validation report: paper's numbers vs. ours, with an explicit note on
   which way each of Section 1's ambiguities (1.6, 1.10) was resolved and
   why (the α=1 test result is the evidence to cite for 1.6)
