"""
Benchmark: jaxtra pentadiagonal_solve vs dense LU vs scipy banded Cholesky
===========================================================================
For an SPD pentadiagonal system A x = b of size n×n:

  jaxtra          : pentadiagonal_solve  (LAPACK gbsv / cuSPARSE gpsvInterleavedBatch)
  dense LU        : jnp.linalg.solve     (O(n^3), dense)
  scipy banded    : scipy.linalg.solveh_banded  (banded Cholesky, O(kn), k=2)

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
from scipy import linalg as scipy_linalg

from jaxtra._src.lax.linalg import pentadiagonal_solve

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
def _dense_solve(A, b):
    """Dense LU solve via jnp.linalg.solve."""
    return jnp.linalg.solve(A, b)


def _scipy_banded_solve(ab_upper, b_np):
    """Banded Cholesky via scipy.linalg.solveh_banded (upper triangular storage)."""
    return scipy_linalg.solveh_banded(ab_upper, b_np, lower=False)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def make_spd_penta(n, dtype=np.float64):
    """Build an SPD pentadiagonal system using the biharmonic stencil.

    Returns (ds, dl, d, du, dw, b, A_dense, ab_upper_scipy) where:
      - ds, dl, d, du, dw are the five diagonals (length n each)
      - b is the right-hand side
      - A_dense is the full n×n matrix (for dense solver)
      - ab_upper_scipy is the banded upper-triangular storage for scipy
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

    return ds_np, dl_np, d_np, du_np, dw_np, b_np, A_dense, ab_upper


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
    ("jaxtra (gbsv)",    "#1f77b4", "o"),
    ("dense LU",         "#ff7f0e", "s"),
    ("scipy banded",     "#2ca02c", "^"),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

records = []

print(f"\n{'n':>7}  {'jaxtra (ms)':>12}  {'dense LU (ms)':>14}  "
      f"{'scipy banded (ms)':>18}  {'vs dense':>9}  {'vs scipy':>9}")
print("-" * 80)

for n in SIZES:
    ds_np, dl_np, d_np, du_np, dw_np, b_np, A_dense, ab_upper = make_spd_penta(n)

    # JAX arrays
    ds_j  = jnp.array(ds_np)
    dl_j  = jnp.array(dl_np)
    d_j   = jnp.array(d_np)
    du_j  = jnp.array(du_np)
    dw_j  = jnp.array(dw_np)
    b_j   = jnp.array(b_np)
    A_j   = jnp.array(A_dense)

    t_jaxtra = time_jax_fn(_jaxtra_solve, ds_j, dl_j, d_j, du_j, dw_j, b_j,
                            n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_dense  = time_jax_fn(_dense_solve, A_j, b_j,
                            n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_scipy  = time_numpy_fn(_scipy_banded_solve, ab_upper, b_np,
                              n_warmup=N_WARMUP, n_repeat=N_REPEAT)

    records.append({"n": n, "method": "jaxtra (gbsv)",  "time_ms": t_jaxtra * 1e3})
    records.append({"n": n, "method": "dense LU",       "time_ms": t_dense  * 1e3})
    records.append({"n": n, "method": "scipy banded",   "time_ms": t_scipy  * 1e3})

    print(f"{n:>7d}  "
          f"{t_jaxtra*1e3:12.2f}  "
          f"{t_dense*1e3:14.2f}  "
          f"{t_scipy*1e3:18.2f}  "
          f"{t_dense/t_jaxtra:9.2f}x  "
          f"{t_scipy/t_jaxtra:9.2f}x")

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
ax.set_title("Pentadiagonal SPD solve  —  jaxtra vs dense LU vs scipy banded", fontsize=11)
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
