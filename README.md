# Time-Fractional Black–Scholes Reproduction

Reproduction of: S.M. Nuugulu, F. Gideon, K.C. Patidar, *"A robust numerical
scheme for a time-fractional Black-Scholes partial differential equation
describing stock exchange dynamics"*, Chaos, Solitons and Fractals 145
(2021) 110753.

**Start here: [plan.md](plan.md).** It records the resolved conventions we
need to agree on before writing code (several places where the paper is
ambiguous or where a literal transcription would be wrong), the module
interface contracts, the build order with stopping tests, and the division
of work between the two of us.

**Then read [docs/validation_report.md](docs/validation_report.md).** The
paper's printed closed-form coefficients for the main time-stepping update
(eqs. 3.11-3.12) turned out to be numerically unstable and inconsistent with
the paper's own underlying discretization; the report documents the
corrected closed form actually implemented in `src/`, the evidence behind
it, and one discrepancy (at alpha=1, against the analytic solution at fixed
grid resolution) that remains open.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running tests

```bash
pytest tests/
```

## Layout

- `src/` — the tfBS solver implementation
- `tests/` — stopping-test suite (see plan.md Section 4 for what each phase must pass)
- `experiments/` — scripts reproducing the paper's examples, figures, and tables
- `results/` — generated figures and tables
- `notebooks/` — final narrative notebook
