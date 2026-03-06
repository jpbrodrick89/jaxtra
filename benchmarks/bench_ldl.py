"""Benchmark: LDL vs LU full solve (factorization + triangular solve).

Measures wall-clock time for the FULL SOLVE (factorization + triangular solve)
of a single 1-D right-hand side vector, comparing:

  - LDL (sytrf) via jaxtra  — for symmetric/Hermitian indefinite matrices
  - LU  (getrf) via jax.scipy.linalg.solve  — always applicable

LDL factorization is JIT-compiled.  The LDL triangular solve is performed in
NumPy (ldl_solve is not yet JIT-traceable), while jax.scipy.linalg.solve is
fully JIT-compiled, so timings are not perfectly apples-to-apples; the total
wall time still reflects end-to-end solve cost for a caller.

Matrix sizes: 50 to 5000.  Two series:
  1. Real symmetric indefinite (f64)
  2. Complex Hermitian indefinite (c128)

Note on D storage: jaxtra returns D as a full (n, n) block-diagonal matrix,
NOT a 1-D array.  Bunch-Kaufman pivoting can produce 2×2 blocks in D, so a
1-D representation is insufficient.

Results are written to benchmarks/results/bench_ldl.csv and companion PNGs.

Usage:
  python benchmarks/bench_ldl.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax._src.lax.linalg import lu as jax_lu

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import ldl as ldl_primitive, ldl_solve

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SIZES = [50, 100, 200, 500, 1000, 2000, 5000]
N_WARMUP = 1
N_REPEAT = 5
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# JIT-compiled kernels (defined once at module level)
# ---------------------------------------------------------------------------


@jax.jit
def _ldl_sym(a):
  """LDL factorisation — real symmetric."""
  return ldl_primitive(a, lower=True, hermitian=False)


@jax.jit
def _ldl_herm(a):
  """LDL factorisation — complex Hermitian."""
  return ldl_primitive(a, lower=True, hermitian=True)


@jax.jit
def _lu(a):
  """LU factorisation (factorization only, for reference)."""
  return jax_lu(a)


@jax.jit
def _lu_solve_real(a, b):
  """LU full solve — real."""
  return jax.scipy.linalg.solve(a, b)


@jax.jit
def _lu_solve_cplx(a, b):
  """LU full solve — complex."""
  return jax.scipy.linalg.solve(a, b)


# ---------------------------------------------------------------------------
# LDL full solve wrappers (JIT factorization + NumPy triangular solve)
# ---------------------------------------------------------------------------


def _ldl_full_sym(a, b):
  """LDL full solve — real symmetric (factorization + solve)."""
  factors, ipiv = _ldl_sym(a)
  return ldl_solve(factors, ipiv, b, lower=True, hermitian=False)


def _ldl_full_herm(a, b):
  """LDL full solve — complex Hermitian (factorization + solve)."""
  factors, ipiv = _ldl_herm(a)
  return ldl_solve(factors, ipiv, b, lower=True, hermitian=True)


# ---------------------------------------------------------------------------
# Timing helper (same pattern as bench_least_squares.py)
# ---------------------------------------------------------------------------


def time_jax_fn(fn, *args, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
  """Warm up then return median wall-time in seconds (block_until_ready)."""
  for _ in range(n_warmup):
    jax.block_until_ready(fn(*args))
  times = []
  for _ in range(n_repeat):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    times.append(time.perf_counter() - t0)
  return float(np.median(times))


# ---------------------------------------------------------------------------
# Matrix factories
# ---------------------------------------------------------------------------


def _sym_indef(n: int, dtype) -> jax.Array:
  """Real symmetric indefinite matrix."""
  a = RNG.standard_normal((n, n)).astype(dtype)
  a = a + a.T
  # Alternating diagonal ensures indefiniteness.
  a += np.diag(np.where(np.arange(n) % 2 == 0, n, -n).astype(dtype))
  return jnp.array(a)


def _herm_indef(n: int, dtype) -> jax.Array:
  """Complex Hermitian indefinite matrix."""
  real_part = np.float32 if dtype == np.complex64 else np.float64
  a = (
    RNG.standard_normal((n, n)).astype(real_part)
    + 1j * RNG.standard_normal((n, n)).astype(real_part)
  ).astype(dtype)
  a = a + np.conj(a).T
  a += np.diag(np.where(np.arange(n) % 2 == 0, n, -n).astype(dtype))
  return jnp.array(a)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _run_series(
  label: str,
  dtype,
  matrix_fn,
  ldl_full_fn,
  lu_solve_fn,
) -> list[dict]:
  rows = []
  print(
    f"  {'n':>6}  {'LDL factoriz (ms)':>18}  {'LDL full solve (ms)':>20}  "
    f"{'LU full solve (ms)':>19}  {'LDL vs LU':>10}"
  )
  print("  " + "-" * 80)
  for n in SIZES:
    a = matrix_fn(n, dtype)
    b = jnp.ones(n, dtype=dtype)

    t_ldl_fact = time_jax_fn(_ldl_sym if "sym" in label else _ldl_herm, a)
    t_ldl_full = time_jax_fn(ldl_full_fn, a, b)
    t_lu_full = time_jax_fn(lu_solve_fn, a, b)

    speedup = t_lu_full / t_ldl_full if t_ldl_full > 0 else float("nan")
    print(
      f"  {n:>6d}  {t_ldl_fact * 1e3:18.2f}  {t_ldl_full * 1e3:20.2f}  "
      f"{t_lu_full * 1e3:19.2f}  {speedup:9.2f}x"
    )

    rows.append({
      "series": label,
      "n": n,
      "ldl_factoriz_ms": t_ldl_fact * 1e3,
      "ldl_solve_ms": t_ldl_full * 1e3,
      "lu_solve_ms": t_lu_full * 1e3,
      "ldl_vs_lu_speedup": speedup,
    })
  return rows


def main():
  all_rows: list[dict] = []

  print("=== Real symmetric indefinite (f64) ===")
  all_rows += _run_series(
    "sym_f64",
    np.float64,
    _sym_indef,
    _ldl_full_sym,
    _lu_solve_real,
  )

  print("\n=== Complex Hermitian indefinite (c128) ===")
  all_rows += _run_series(
    "herm_c128",
    np.complex128,
    _herm_indef,
    _ldl_full_herm,
    _lu_solve_cplx,
  )

  # Write CSV.
  csv_path = RESULTS_DIR / "bench_ldl.csv"
  fieldnames = [
    "series",
    "n",
    "ldl_factoriz_ms",
    "ldl_solve_ms",
    "lu_solve_ms",
    "ldl_vs_lu_speedup",
  ]
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
  print(f"\nResults written to {csv_path}")

  _plot(all_rows)


def _plot(rows: list[dict]):
  series_configs = [
    ("sym_f64", "Real symmetric indefinite (f64)"),
    ("herm_c128", "Complex Hermitian indefinite (c128)"),
  ]

  fig, axes = plt.subplots(1, 2, figsize=(13, 5))

  for ax, (series, title) in zip(axes, series_configs):
    data = [r for r in rows if r["series"] == series]
    ns = [r["n"] for r in data]

    ax.loglog(
      ns,
      [r["ldl_solve_ms"] for r in data],
      "o-",
      label="LDL full solve (sytrf + trsv)",
      linewidth=2,
      markersize=5,
    )
    ax.loglog(
      ns,
      [r["lu_solve_ms"] for r in data],
      "s--",
      label="LU full solve (scipy.linalg.solve)",
      linewidth=2,
      markersize=5,
    )
    ax.loglog(
      ns,
      [r["ldl_factoriz_ms"] for r in data],
      "^:",
      label="LDL factorization only",
      linewidth=1.5,
      markersize=4,
      alpha=0.6,
    )

    ax.set_xlabel("Matrix size n", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

  fig.suptitle(
    "Full solve = factorization + triangular solve, 1-D rhs", fontsize=11
  )
  fig.tight_layout()
  png_path = RESULTS_DIR / "bench_ldl.png"
  fig.savefig(png_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Plot saved to {png_path}")


if __name__ == "__main__":
  main()
