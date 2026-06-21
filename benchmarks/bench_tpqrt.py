"""
Benchmark: jaxtra tpqrt (triangular-pentagonal QR) vs raw geqrf
===============================================================
The trust-region Levenberg-Marquardt step re-triangularises

      C = [ R ]      R : (n, n) upper triangular factor of the Jacobian
          [ D ]      D : (n, n) diagonal regularisation

into a new factor R̃ (with R̃ᵀR̃ = RᵀR + DᵀD) plus the Householder reflectors
``V, T``.  The two backends of the ``tpqrt`` primitive are compared against a
plain QR baseline:

  jaxtra tpqrt (LAPACK FFI) : tpqrt(R, D, l=n) on CPU — LAPACK ?tpqrt, exploits
                           both the triangular R and the triangular (diagonal) D
  jaxtra tpqrt (blocked HH) : the pure-JAX fallback used off-CPU / under autodiff
                           — compact-WY blocked Householder (BLAS-3 trailing
                           updates), returning the same (R, V, T)
  raw geqrf              : jax._src.lax.linalg.geqrf on the dense (2n, n) stack
                           (a full QR; ignores the triangular structure)

Note: on CPU the primitive uses the LAPACK FFI path, which beats geqrf; the
blocked-Householder line is the fallback for platforms without a tpqrt FFI
target (e.g. TPU) and for differentiation.

Results are written to  benchmarks/results/bench_tpqrt.csv
Plot is written to      benchmarks/results/bench_tpqrt.png

Usage:
  python benchmarks/bench_tpqrt.py
"""

import csv
import functools
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax._src.lax.linalg import geqrf
from jaxtra._src.lax.linalg import tpqrt, _tpqrt_blocked_householder_2d

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# JIT-compiled re-triangularisations (n is static => l baked into the trace).
# Each returns the full factorization so XLA cannot prune V / T.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _lapack_fn(n):
    return jax.jit(lambda R, D: tpqrt(R, D, l=n))           # LAPACK FFI


@functools.lru_cache(maxsize=None)
def _blocked_fn(n, nb=32):
    return jax.jit(lambda R, D: _tpqrt_blocked_householder_2d(R, D, n, nb))


@functools.lru_cache(maxsize=None)
def _geqrf_fn(n):
    def f(R, D):
        C = jnp.concatenate([R, D], axis=0)
        return geqrf(C)                                     # (qr, taus)
    return jax.jit(f)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def make_problem(n, dtype=np.float64):
    """An upper triangular R and a diagonal regularisation D (both n×n)."""
    rng = np.random.default_rng(0)
    R = np.triu(rng.standard_normal((n, n))).astype(dtype)
    # Strengthen the diagonal so R is well-conditioned (as for a real factor).
    R[np.diag_indices(n)] += n
    d = np.abs(rng.standard_normal(n)).astype(dtype) + 0.5
    D = np.diag(d).astype(dtype)
    return jnp.asarray(R), jnp.asarray(D)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_jax_fn(fn, *args, n_warmup=3, n_repeat=10):
    """Warm up then return median wall-time in seconds."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SIZES = [16, 32, 64, 128, 256, 512, 1024]
N_WARMUP = 3
N_REPEAT = 10

METHODS = [
    ("jaxtra tpqrt (LAPACK FFI)",      "#1f77b4", "o"),
    ("jaxtra tpqrt (blocked HH)",      "#8c564b", "P"),
    ("raw geqrf",                      "#d62728", "s"),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

records = []

print(f"\n{'n':>6}  {'LAPACK FFI':>11}  {'blocked HH':>11}  {'geqrf':>11}  "
      f"{'LAPACK/qr':>10}  {'blkHH/qr':>9}")
print(f"{'':6}  {'(ms)':>11}  {'(ms)':>11}  {'(ms)':>11}")
print("-" * 70)

for n in SIZES:
    R, D = make_problem(n)

    f_lapack = _lapack_fn(n)
    f_blocked = _blocked_fn(n)
    f_geqrf = _geqrf_fn(n)

    # Sanity check: R factors agree (sign/phase-invariant via the Gram matrix).
    g = lambda M: M.conj().T @ M
    ref = g(jnp.triu(f_geqrf(R, D)[0][:n]))
    for f in (f_lapack, f_blocked):
        assert float(jnp.max(jnp.abs(g(f(R, D)[0]) - ref))) < 1e-7

    t_lapack = time_jax_fn(f_lapack, R, D, n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_blocked = time_jax_fn(f_blocked, R, D, n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_geqrf = time_jax_fn(f_geqrf, R, D, n_warmup=N_WARMUP, n_repeat=N_REPEAT)

    records.append({"n": n, "method": "jaxtra tpqrt (LAPACK FFI)", "time_ms": t_lapack * 1e3})
    records.append({"n": n, "method": "jaxtra tpqrt (blocked HH)", "time_ms": t_blocked * 1e3})
    records.append({"n": n, "method": "raw geqrf",                 "time_ms": t_geqrf * 1e3})

    print(f"{n:>6d}  {t_lapack*1e3:11.3f}  {t_blocked*1e3:11.3f}  "
          f"{t_geqrf*1e3:11.3f}  {t_lapack/t_geqrf:9.2f}x  {t_blocked/t_geqrf:8.2f}x")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "bench_tpqrt.csv"
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

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("System size n", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("Trust-region LM update  [R; D] → R̃  —  jaxtra tpqrt vs raw geqrf",
             fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=10)
fig.tight_layout()

png_path = RESULTS_DIR / "bench_tpqrt.png"
fig.savefig(png_path, dpi=130)
print(f"Plot saved to {png_path}")
