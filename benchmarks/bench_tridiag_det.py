"""
Benchmark: tridiagonal determinant — scan-based recurrence vs LAPACK gttrf
===========================================================================

Compares two approaches to computing log|det(A)| for a random tridiagonal
matrix A of size n×n:

  scan   : three-term recurrence via lax.scan (unroll=8), O(n) sequential
  gttrf  : LAPACK gttrf tridiagonal LU, then prod(U diagonal) + pivot sign

Both unbatched (single matrix) and batched (vmap over a batch of matrices)
are benchmarked across n = 64, 128, 256, 512, 1024, 2048, 4096.

Results are written to  benchmarks/results/bench_tridiag_det.csv
Plot is  written to     benchmarks/results/bench_tridiag_det.png

Usage:
    uv run --locked python benchmarks/bench_tridiag_det.py
"""

import csv
import pathlib
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import tridiag_logdet_lu

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Scan-based reference (from the docstring / original implementation)
# ---------------------------------------------------------------------------

@jax.jit
def tridiag_logdet_scan(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Numerically stable log-det via signed log-space three-term recurrence."""
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, i):
        log_f2, sgn_f2, log_f1, sgn_f1 = carry

        log_t1 = jnp.log(jnp.abs(b[i])) + log_f1
        sgn_t1 = jnp.sign(b[i]) * sgn_f1

        log_t2 = jnp.log(jnp.abs(coupling[i])) + log_f2
        sgn_t2 = jnp.sign(coupling[i]) * sgn_f2

        log_max = jnp.maximum(log_t1, log_t2)
        val = sgn_t1 * jnp.exp(log_t1 - log_max) - sgn_t2 * jnp.exp(log_t2 - log_max)
        log_f = log_max + jnp.log(jnp.abs(val))
        sgn_f = jnp.sign(val)

        return (log_f1, sgn_f1, log_f, sgn_f), None

    log_b0 = jnp.log(jnp.abs(b[0]))
    sgn_b0 = jnp.sign(b[0])
    init = (0.0, 1.0, log_b0, sgn_b0)
    (_, _, log_det, sgn_det), _ = lax.scan(
        step, init, jnp.arange(1, b.shape[0]), unroll=8
    )
    return sgn_det, log_det


tridiag_logdet_scan_batched = jax.jit(jax.vmap(tridiag_logdet_scan))
tridiag_logdet_lu_batched   = jax.jit(jax.vmap(tridiag_logdet_lu))

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_jax_fn(fn, *args, n_warmup=3, n_repeat=10):
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(n=128):
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n) + 4.0
    a = rng.standard_normal(n - 1) * 0.5
    c = rng.standard_normal(n - 1) * 0.5
    A = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
    _, ref_ldet = jnp.linalg.slogdet(jnp.array(A))

    s_scan, l_scan = tridiag_logdet_scan(
        jnp.array(a), jnp.array(b), jnp.array(c))
    s_lu,   l_lu   = tridiag_logdet_lu(
        jnp.array(a), jnp.array(b), jnp.array(c))

    errs = {
        "scan": abs(float(l_scan) - float(ref_ldet)),
        "gttrf": abs(float(l_lu)  - float(ref_ldet)),
    }
    for name, err in errs.items():
        status = "OK" if err < 1e-9 else f"FAIL (err={err:.2e})"
        print(f"  correctness [{name:5s}]: {status}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES   = [64, 128, 256, 512, 1024, 2048, 4096]
BATCH   = 64      # batch size for batched benchmark
RNG     = np.random.default_rng(42)

METHODS = [
    ("scan (unbatched)",  "#1f77b4", "o", "-"),
    ("gttrf (unbatched)", "#ff7f0e", "s", "-"),
    ("scan (batched)",    "#1f77b4", "o", "--"),
    ("gttrf (batched)",   "#ff7f0e", "s", "--"),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("Correctness check (n=128):")
check_correctness(128)

print(f"\nBenchmark  (batch={BATCH})\n")
print(f"{'n':>6}  {'scan-ub (ms)':>13}  {'gttrf-ub (ms)':>14}  "
      f"{'speedup-ub':>11}  {'scan-b (ms)':>12}  {'gttrf-b (ms)':>13}  {'speedup-b':>10}")
print("  " + "-" * 90)

records = []

for n in SIZES:
    # Unbatched inputs
    b  = jnp.array(RNG.standard_normal(n) + 4.0)
    a  = jnp.array(RNG.standard_normal(n - 1) * 0.5)
    c  = jnp.array(RNG.standard_normal(n - 1) * 0.5)

    # Batched inputs (BATCH × n)
    B_b  = jnp.array(RNG.standard_normal((BATCH, n)) + 4.0)
    A_b  = jnp.array(RNG.standard_normal((BATCH, n - 1)) * 0.5)
    C_b  = jnp.array(RNG.standard_normal((BATCH, n - 1)) * 0.5)

    t_scan_ub  = time_jax_fn(jax.jit(tridiag_logdet_scan), a, b, c)
    t_gttrf_ub = time_jax_fn(jax.jit(tridiag_logdet_lu),   a, b, c)
    t_scan_b   = time_jax_fn(tridiag_logdet_scan_batched,   A_b, B_b, C_b)
    t_gttrf_b  = time_jax_fn(tridiag_logdet_lu_batched,     A_b, B_b, C_b)

    sp_ub = t_scan_ub / t_gttrf_ub
    sp_b  = t_scan_b  / t_gttrf_b

    print(f"{n:>6d}  {t_scan_ub*1e3:13.3f}  {t_gttrf_ub*1e3:14.3f}  "
          f"{sp_ub:11.2f}x  {t_scan_b*1e3:12.3f}  {t_gttrf_b*1e3:13.3f}  {sp_b:9.2f}x")

    for method, ms in [
        ("scan (unbatched)",  t_scan_ub  * 1e3),
        ("gttrf (unbatched)", t_gttrf_ub * 1e3),
        ("scan (batched)",    t_scan_b   * 1e3),
        ("gttrf (batched)",   t_gttrf_b  * 1e3),
    ]:
        records.append({"n": n, "method": method, "time_ms": ms})

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "bench_tridiag_det.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["n", "method", "time_ms"])
    writer.writeheader()
    writer.writerows(records)
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
titles = ["Unbatched (single matrix)", f"Batched (batch={BATCH})"]
unbatched_methods = ["scan (unbatched)", "gttrf (unbatched)"]
batched_methods   = ["scan (batched)",   "gttrf (batched)"]

for ax, title, methods_in_ax in zip(
    axes, titles, [unbatched_methods, batched_methods]
):
    for label, color, marker, ls in METHODS:
        if label not in methods_in_ax:
            continue
        pts = [(r["n"], r["time_ms"]) for r in records if r["method"] == label]
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, label=label.replace(" (unbatched)", "").replace(" (batched)", ""),
                color=color, marker=marker, linestyle=ls,
                linewidth=2, markersize=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size n", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(SIZES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

fig.suptitle("Tridiagonal log-det: scan recurrence vs LAPACK gttrf", fontsize=14)
fig.tight_layout()
out = RESULTS_DIR / "bench_tridiag_det.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved to {out}")

print("\nSpeedup = time(scan) / time(gttrf) — > 1 means gttrf is faster")
