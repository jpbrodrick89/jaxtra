"""
Benchmark: jaxtra ORMQR vs dense QR vs jax lstsq (SVD)
=======================================================
For an overdetermined system A @ x ≈ b:

  jaxtra     : geqrf(A) + ormqr(H, taus, b) + solve_triangular  (Q never formed)
  dense QR   : jnp.linalg.qr(A) + Q.T @ b + solve_triangular    (Q materialised)
  jax lstsq  : jax.numpy.linalg.lstsq  (SVD-based, JAX arrays)
  scipy gelsy: scipy.linalg.lstsq(lapack_driver="gelsy")  (CPU cols only: not 200 or 512)

Results are written to  benchmarks/results/bench_least_squares[_gpu].csv
Plots are written to    benchmarks/results/bench_cols{20,50,100}[_gpu].png

Usage:
  python benchmarks/bench_least_squares.py           # CPU run
  python benchmarks/bench_least_squares.py --gpu     # GPU run with larger sizes
"""

import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true",
                    help="GPU mode: also benchmark at 50k and 100k rows, "
                         "append '_gpu' to output filenames.")
args = parser.parse_args()

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUFFIX = "_gpu" if args.gpu else ""

# ---------------------------------------------------------------------------
# JIT-compiled solvers
# ---------------------------------------------------------------------------

@jax.jit
def _jaxtra_solve(A, b):
    """geqrf + ormqr + triangular solve — Q never materialised."""
    Qtb, R = qr_multiply(A, b, mode='right')
    return jsl.solve_triangular(R, Qtb)


@jax.jit
def _dense_qr_solve(A, b):
    """geqrf + orgqr + GEMM + triangular solve — Q materialised."""
    Q, R = jnp.linalg.qr(A)
    Qtb = Q.T @ b
    return jsl.solve_triangular(R, Qtb)


@jax.jit
def _jax_lstsq(A, b):
    """jax.numpy.linalg.lstsq (SVD-based)."""
    x, _, _, _ = jnp.linalg.lstsq(A, b)
    return x


def _scipy_gelsy_solve(A_np, b_np):
    """scipy.linalg.lstsq with GELSY driver (QR with column pivoting)."""
    x, _, _, _ = scipy_linalg.lstsq(A_np, b_np, lapack_driver="gelsy")
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
if args.gpu:
    COL_COUNTS = COL_COUNTS + [200, 512]
    ROW_SIZES  = ROW_SIZES + [50_000, 100_000, 200_000]

# Columns for which scipy gelsy (CPU) is too slow / not applicable on GPU
GPU_ONLY_COLS = {200, 512}

N_WARMUP = 5  if args.gpu else 1
N_REPEAT = 20 if args.gpu else 5
RNG = np.random.default_rng(0)

METHODS = [
    ("jaxtra (ORMQR)", "#1f77b4", "o"),
    ("dense QR",       "#ff7f0e", "s"),
    ("jax lstsq",      "#2ca02c", "^"),
    ("scipy gelsy",    "#9467bd", "D"),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

records = []   # list of dicts written to CSV

for n_cols in COL_COUNTS:
    run_gelsy = n_cols not in GPU_ONLY_COLS
    print(f"\nn_cols = {n_cols}")
    hdr = (f"  {'n_rows':>7}  {'jaxtra (ms)':>12}  {'dense QR (ms)':>14}  "
           f"{'jax lstsq (ms)':>15}")
    if run_gelsy:
        hdr += f"  {'scipy gelsy (ms)':>17}"
    hdr += f"  {'vs QR':>6}  {'vs lstsq':>9}"
    print(hdr)
    print("  " + "-" * (78 + (19 if run_gelsy else 0)))

    for n_rows in ROW_SIZES:
        if n_rows < n_cols:
            continue
        A_np = RNG.standard_normal((n_rows, n_cols)).astype(np.float64)
        b_np = RNG.standard_normal(n_rows).astype(np.float64)
        A_jx = jax.block_until_ready(jax.device_put(A_np))
        b_jx = jax.block_until_ready(jax.device_put(b_np))

        t_jaxtra = time_jax_fn(_jaxtra_solve, A_jx, b_jx,
                                n_warmup=N_WARMUP, n_repeat=N_REPEAT)
        t_dense  = time_jax_fn(_dense_qr_solve, A_jx, b_jx,
                                n_warmup=N_WARMUP, n_repeat=N_REPEAT)
        t_lstsq  = time_jax_fn(_jax_lstsq, A_jx, b_jx,
                                n_warmup=N_WARMUP, n_repeat=N_REPEAT)

        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "jaxtra (ORMQR)", "time_ms": t_jaxtra * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "dense QR",       "time_ms": t_dense  * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "jax lstsq",      "time_ms": t_lstsq  * 1e3})

        row = (f"  {n_rows:>7d}  "
               f"{t_jaxtra*1e3:12.2f}  "
               f"{t_dense*1e3:14.2f}  "
               f"{t_lstsq*1e3:15.2f}")

        if run_gelsy:
            t_gelsy = time_numpy_fn(_scipy_gelsy_solve, A_np, b_np,
                                     n_warmup=N_WARMUP, n_repeat=N_REPEAT)
            records.append({"n_rows": n_rows, "n_cols": n_cols,
                             "method": "scipy gelsy", "time_ms": t_gelsy * 1e3})
            row += f"  {t_gelsy*1e3:17.2f}"
            gelsy_ratio = f"{t_gelsy/t_jaxtra:8.2f}x"
        else:
            gelsy_ratio = ""

        row += (f"  {t_dense/t_jaxtra:6.2f}x"
                f"  {t_lstsq/t_jaxtra:8.2f}x")
        print(row)

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / f"bench_least_squares{SUFFIX}.csv"
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
        if not pts:
            continue
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, label=label, color=color, marker=marker,
                linewidth=2, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of rows (M)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Least-squares solve  —  {n_cols} columns", fontsize=13)
    ax.set_xticks(ROW_SIZES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out = RESULTS_DIR / f"bench_cols{n_cols}{SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")

print("\nSpeedup = time(reference) / time(jaxtra)  — higher is better for jaxtra")
