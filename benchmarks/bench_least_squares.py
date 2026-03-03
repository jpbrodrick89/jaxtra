"""
Benchmark: jaxtra ORMQR vs dense QR vs scipy gelsy vs JAX lstsq
================================================================
For an overdetermined system A @ x ≈ b where b is a 1-D vector (single RHS):

  jaxtra (ORMQR) : geqrf(A) + ormqr(H, taus, b) + solve_triangular  (Q never formed)
  dense QR       : jnp.linalg.qr(A) + Q.T @ b + solve_triangular    (Q materialised)
  scipy gelsy    : scipy.linalg.lstsq(lapack_driver='gelsy')          (QR-based)
  JAX lstsq      : jnp.linalg.lstsq                                   (SVD-based)

All benchmarks use a single right-hand side (b is always a 1-D vector of length
n_rows).  The three subplot titles label the number of matrix columns (n_cols),
i.e. the number of unknowns in the least-squares problem.

Performance note
----------------
jaxtra avoids materialising Q entirely; its Q-apply cost is O(m * n_cols).
Dense QR must first form Q via dorgqr at O(m * n_cols^2) cost, then apply it.
As a result, jaxtra's advantage over dense QR grows with n_cols (matrix columns):
with few columns the Q-formation cost is small and dense QR can be competitive,
but with more columns the O(m * n_cols^2) term dominates and jaxtra pulls ahead.
This has nothing to do with the number of RHS columns (always 1 here); the
relevant dimension is n_cols (columns of A).

Results are written to  benchmarks/results/bench_least_squares.csv
Plots are written to    benchmarks/results/bench_cols{20,50,100}.png

Usage:
  python benchmarks/bench_least_squares.py
"""

import csv
import pathlib
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from scipy import linalg as scipy_linalg

from jaxtra.scipy.linalg import qr_multiply

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# JIT-compiled solvers
# ---------------------------------------------------------------------------

@jax.jit
def _jaxtra_solve(A, b):
    """Least-squares via ORMQR (Q never materialised)."""
    Qtb, R = qr_multiply(A, b, mode='right')
    return jsl.solve_triangular(R, Qtb)


@jax.jit
def _dense_qr_solve(A, b):
    """Least-squares via dense QR (Q explicitly formed)."""
    Q, R = jnp.linalg.qr(A)
    Qtb = Q.T @ b
    return jsl.solve_triangular(R, Qtb)


@jax.jit
def _jax_lstsq_solve(A, b):
    """Least-squares via jnp.linalg.lstsq (SVD-based)."""
    x, _, _, _ = jnp.linalg.lstsq(A, b)
    return x


def _scipy_gelsy_solve(A_np, b_np):
    """Least-squares via scipy.linalg.lstsq with gelsy driver (QR-based)."""
    x, _, _, _ = scipy_linalg.lstsq(A_np, b_np, lapack_driver='gelsy')
    return x


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_jax_fn(fn, *args, n_warmup=1, n_repeat=5):
    """Warm up then return median wall-time in seconds (block_until_ready)."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_numpy_fn(fn, *args, n_warmup=1, n_repeat=5):
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

COL_COUNTS = [20, 50, 100]
ROW_SIZES  = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]

N_WARMUP = 1
N_REPEAT = 5
RNG = np.random.default_rng(0)

METHODS = [
    ("jaxtra (ORMQR)", "#1f77b4", "o"),
    ("dense QR",       "#ff7f0e", "s"),
    ("scipy gelsy",    "#2ca02c", "^"),
    ("JAX lstsq",      "#d62728", "D"),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

records = []   # list of dicts written to CSV

for n_cols in COL_COUNTS:
    print(f"\nn_cols = {n_cols}  (matrix columns / unknowns; b is always 1-D)")
    print(f"  {'n_rows':>7}  {'jaxtra (ms)':>12}  {'dense QR (ms)':>14}  "
          f"{'gelsy (ms)':>11}  {'JAX lstsq (ms)':>15}")
    print("  " + "-" * 68)

    for n_rows in ROW_SIZES:
        A_np = RNG.standard_normal((n_rows, n_cols)).astype(np.float64)
        b_np = RNG.standard_normal(n_rows).astype(np.float64)
        A_jx = jnp.array(A_np)
        b_jx = jnp.array(b_np)

        t_jaxtra  = time_jax_fn(_jaxtra_solve,    A_jx, b_jx,
                                 n_warmup=N_WARMUP, n_repeat=N_REPEAT)
        t_dense   = time_jax_fn(_dense_qr_solve,  A_jx, b_jx,
                                 n_warmup=N_WARMUP, n_repeat=N_REPEAT)
        t_gelsy   = time_numpy_fn(_scipy_gelsy_solve, A_np, b_np,
                                   n_warmup=N_WARMUP, n_repeat=N_REPEAT)
        t_jlstsq  = time_jax_fn(_jax_lstsq_solve, A_jx, b_jx,
                                 n_warmup=N_WARMUP, n_repeat=N_REPEAT)

        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "jaxtra (ORMQR)", "time_ms": t_jaxtra  * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "dense QR",        "time_ms": t_dense   * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "scipy gelsy",     "time_ms": t_gelsy   * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "JAX lstsq",       "time_ms": t_jlstsq  * 1e3})

        print(f"  {n_rows:>7d}  "
              f"{t_jaxtra*1e3:12.2f}  "
              f"{t_dense*1e3:14.2f}  "
              f"{t_gelsy*1e3:11.2f}  "
              f"{t_jlstsq*1e3:15.2f}")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "bench_least_squares.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["n_rows", "n_cols", "method", "time_ms"])
    writer.writeheader()
    writer.writerows(records)
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Plots — one per column count
# ---------------------------------------------------------------------------

for n_cols in COL_COUNTS:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, color, marker in METHODS:
        pts = [(r["n_rows"], r["time_ms"])
               for r in records
               if r["n_cols"] == n_cols and r["method"] == label]
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, label=label, color=color, marker=marker,
                linewidth=2, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of rows (M)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Least-squares solve  —  {n_cols} matrix columns, 1 RHS", fontsize=13)
    ax.set_xticks(ROW_SIZES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out = RESULTS_DIR / f"bench_cols{n_cols}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")

print("\nSpeedup = time(reference) / time(jaxtra)  — higher is better for jaxtra")
