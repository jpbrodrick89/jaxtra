"""
Benchmark: jaxtra pentadiagonal_solve vs dense solvers vs scipy banded
=======================================================================
For an SPD pentadiagonal system A x = b of size n×n:

  jaxtra gbsv         : pentadiagonal_solve   (LAPACK gbsv / cuSPARSE gpsvInterleavedBatch)
  jaxtra pbsv         : pentadiagonal_solveh  (LAPACK pbsv — banded Cholesky, O(kn), k=2)
  dense LU            : jnp.linalg.solve      (O(n^3), dense)
  JAX Cholesky        : jax.scipy.linalg.cho_factor + cho_solve  (O(n^3), dense)
  scipy banded        : scipy.linalg.solveh_banded  (banded Cholesky, O(kn), k=2)
  scipy solve_banded  : scipy.linalg.solve_banded   (general banded LU, O(kn), k=2)

The SPD pentadiagonal matrix uses the biharmonic stencil:
    A[i, i±2] = 1,  A[i, i±1] = -4,  A[i, i] = (8 + epsilon)
with epsilon = 0.1 to ensure strict positive definiteness.

Results are written to  benchmarks/results/bench_banded.csv
Plot is written to      benchmarks/results/bench_banded.png

Usage:
  python benchmarks/bench_banded.py
"""

import csv
import pathlib
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jax_linalg
from scipy import linalg as scipy_linalg

from jaxtra._src.lax.linalg import pentadiagonal_solve, pentadiagonal_solveh

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# JIT-compiled solvers
# ---------------------------------------------------------------------------

@jax.jit
def _jaxtra_solve(ds, dl, d, du, dw, b):
    """Pentadiagonal solve via jaxtra (LAPACK gbsv / cuSPARSE)."""
    return pentadiagonal_solve(ds, dl, d, du, dw, b)


@jax.jit
def _jaxtra_solveh(d, du, dw, b):
    """Hermitian pentadiagonal solve via jaxtra (LAPACK pbsv)."""
    return pentadiagonal_solveh(d, du, dw, b)


@jax.jit
def _dense_solve(A, b):
    """Dense LU solve via jnp.linalg.solve."""
    return jnp.linalg.solve(A, b)


@jax.jit
def _jax_cholesky_solve(A, b):
    """Dense Cholesky solve via jax.scipy.linalg (cho_factor + cho_solve)."""
    c, lower = jax_linalg.cho_factor(A)
    return jax_linalg.cho_solve((c, lower), b)


def _scipy_banded_solve(ab_upper, b_np):
    """Banded Cholesky via scipy.linalg.solveh_banded (upper triangular storage)."""
    return scipy_linalg.solveh_banded(ab_upper, b_np, lower=False)


def _scipy_solve_banded(ab_banded, b_np):
    """General banded LU solve via scipy.linalg.solve_banded (kl=2, ku=2)."""
    return scipy_linalg.solve_banded((2, 2), ab_banded, b_np)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def make_spd_penta(n, dtype=np.float64):
    """Build an SPD pentadiagonal system using the biharmonic stencil.

    Returns (ds, dl, d, du, dw, b, A_dense, ab_upper_scipy, ab_banded) where:
      - ds, dl, d, du, dw are the five diagonals (length n each)
      - b is the right-hand side
      - A_dense is the full n×n matrix (for dense solvers)
      - ab_upper_scipy is the banded upper-triangular storage for solveh_banded
      - ab_banded is the general banded storage (kl=2, ku=2) for solve_banded
    """
    eps = 0.1  # small shift to ensure strict positive definiteness
    ds_np = np.ones(n, dtype=dtype)
    dl_np = np.full(n, -4.0, dtype=dtype)
    d_np  = np.full(n, 8.0 + eps, dtype=dtype)
    du_np = np.full(n, -4.0, dtype=dtype)
    dw_np = np.ones(n, dtype=dtype)
    rng   = np.random.default_rng(42)
    b_np  = rng.standard_normal(n).astype(dtype)

    # Dense matrix for jnp.linalg.solve reference
    A_dense = np.diag(d_np)
    for i in range(1, n):
        A_dense[i, i - 1] += dl_np[i]
    for i in range(2, n):
        A_dense[i, i - 2] += ds_np[i]
    for i in range(n - 1):
        A_dense[i, i + 1] += du_np[i]
    for i in range(n - 2):
        A_dense[i, i + 2] += dw_np[i]

    # scipy solveh_banded upper storage: ab[0, j] = A[j-2, j], ab[1, j] = A[j-1, j], ab[2, j] = A[j, j]
    # (bandwidth = 2, so 3 rows: superdiagonal+2, superdiagonal+1, main)
    ab_upper = np.zeros((3, n), dtype=dtype)
    ab_upper[2, :] = d_np            # main diagonal
    ab_upper[1, 1:] = du_np[:-1]    # first super-diagonal (A[j, j+1] stored at ab[1, j+1])
    ab_upper[0, 2:] = dw_np[:-2]    # second super-diagonal (A[j, j+2] stored at ab[0, j+2])

    # scipy solve_banded general banded storage (kl=2, ku=2):
    # ab[ku + i - j, j] = A[i, j]  =>  ab[2 + i - j, j] = A[i, j]
    ab_banded = np.zeros((5, n), dtype=dtype)
    ab_banded[0, 2:] = dw_np[:-2]   # row 0 = ku-2: A[j-2, j] = dw[j-2]
    ab_banded[1, 1:] = du_np[:-1]   # row 1 = ku-1: A[j-1, j] = du[j-1]
    ab_banded[2, :]  = d_np          # row 2 = ku:   A[j,   j] = d[j]
    ab_banded[3, :-1] = dl_np[1:]   # row 3 = ku+1: A[j+1, j] = dl[j+1]
    ab_banded[4, :-2] = ds_np[2:]   # row 4 = ku+2: A[j+2, j] = ds[j+2]

    return ds_np, dl_np, d_np, du_np, dw_np, b_np, A_dense, ab_upper, ab_banded


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_jax_fn(fn, *args, n_warmup=2, n_repeat=5):
    """Warm up then return median wall-time in seconds."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_numpy_fn(fn, *args, n_warmup=2, n_repeat=5):
    """Warm up then return median wall-time in seconds."""
    for _ in range(n_warmup):
        fn(*args)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES    = [100, 200, 500, 1000, 2000, 5000, 10000]
N_WARMUP = 2
N_REPEAT = 5

METHODS = [
    ("pentadiagonal_solve",       "#1f77b4", "o"),
    ("pentadiagonal_solveh",       "#17becf", "p"),
    ("dense LU",            "#ff7f0e", "s"),
    ("JAX Cholesky",        "#9467bd", "D"),
    ("scipy solveh_banded", "#2ca02c", "^"),
    ("scipy solve_banded",  "#d62728", "v"),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

records = []

print(f"\n{'n':>7}  {'penta (ms)':>10}  {'pentah (ms)':>10}  {'dense LU (ms)':>14}  "
      f"{'JAX Chol (ms)':>14}  {'solveh_bnd (ms)':>16}  {'solve_bnd (ms)':>15}  "
      f"{'pentah/solveh':>14}")
print("-" * 120)

for n in SIZES:
    ds_np, dl_np, d_np, du_np, dw_np, b_np, A_dense, ab_upper, ab_banded = make_spd_penta(n)

    # JAX arrays
    ds_j  = jnp.array(ds_np)
    dl_j  = jnp.array(dl_np)
    d_j   = jnp.array(d_np)
    du_j  = jnp.array(du_np)
    dw_j  = jnp.array(dw_np)
    b_j   = jnp.array(b_np)
    A_j   = jnp.array(A_dense)

    # pentadiagonal_solve requires rank-2 RHS (n, k)
    b_j_2d = b_j[:, None]

    t_jaxtra    = time_jax_fn(_jaxtra_solve, ds_j, dl_j, d_j, du_j, dw_j, b_j_2d,
                               n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_pbsv      = time_jax_fn(_jaxtra_solveh, d_j, du_j, dw_j, b_j_2d,
                               n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_dense     = time_jax_fn(_dense_solve, A_j, b_j,
                               n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_jax_chol  = time_jax_fn(_jax_cholesky_solve, A_j, b_j,
                               n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_solveh    = time_numpy_fn(_scipy_banded_solve, ab_upper, b_np,
                                n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_solve_bnd = time_numpy_fn(_scipy_solve_banded, ab_banded, b_np,
                                n_warmup=N_WARMUP, n_repeat=N_REPEAT)

    records.append({"n": n, "method": "pentadiagonal_solve",       "time_ms": t_jaxtra    * 1e3})
    records.append({"n": n, "method": "pentadiagonal_solveh",       "time_ms": t_pbsv      * 1e3})
    records.append({"n": n, "method": "dense LU",            "time_ms": t_dense     * 1e3})
    records.append({"n": n, "method": "JAX Cholesky",        "time_ms": t_jax_chol  * 1e3})
    records.append({"n": n, "method": "scipy solveh_banded", "time_ms": t_solveh    * 1e3})
    records.append({"n": n, "method": "scipy solve_banded",  "time_ms": t_solve_bnd * 1e3})

    print(f"{n:>7d}  "
          f"{t_jaxtra*1e3:10.2f}  "
          f"{t_pbsv*1e3:10.2f}  "
          f"{t_dense*1e3:14.2f}  "
          f"{t_jax_chol*1e3:14.2f}  "
          f"{t_solveh*1e3:16.2f}  "
          f"{t_solve_bnd*1e3:15.2f}  "
          f"{t_solveh/t_pbsv:12.2f}x")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "bench_banded.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["n", "method", "time_ms"])
    writer.writeheader()
    writer.writerows(records)
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4.5))

for label, color, marker in METHODS:
    pts = [(r["n"], r["time_ms"]) for r in records if r["method"] == label]
    xs, ys = zip(*sorted(pts))
    ax.plot(xs, ys, label=label, color=color, marker=marker,
            linewidth=2, markersize=5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("System size n", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("Pentadiagonal SPD solve  —  jaxtra vs dense LU / Cholesky vs scipy banded", fontsize=11)
ax.set_xticks(SIZES)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.tick_params(axis="x", labelrotation=45)
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
fig.tight_layout()

out = RESULTS_DIR / "bench_banded.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to {out}")

print("\nSpeedup = time(reference) / time(jaxtra)  — higher is better for jaxtra")
