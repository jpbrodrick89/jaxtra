"""Benchmark: LDL vs LU full solve (factorization + triangular solve).

Measures wall-clock time for the FULL SOLVE (factorization + triangular solve)
of a single 1-D right-hand side vector, comparing:

  - LDL (sytrf) via jaxtra  — for symmetric/Hermitian indefinite matrices
  - LU  (getrf) via jax.scipy.linalg.solve  — always applicable

Columns reported:
  1. sytrf (ms)        — raw LAPACK/cuSolver sytrf/hetrf call only (ldl_raw)
  2. ldl factor (ms)   — ldl() = sytrf + BK untangle scan
  3. ldl full (ms)     — ldl() + ldl_solve() end-to-end
  4. LU full (ms)      — jax.scipy.linalg.solve for reference
  5. ldl_vs_lu         — LU time / LDL full time (> 1 means LDL is faster)

Matrix sizes: 50 to 5000.  Two series:
  1. Real symmetric indefinite (f64)
  2. Complex Hermitian indefinite (c128)

Results are written to benchmarks/results/bench_ldl.csv and a companion PNG.

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

from jaxtra._src.lax.linalg import ldl as ldl_primitive, ldl_solve, ldl_raw

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
def _sytrf_sym(a):
  """Raw sytrf call only — real symmetric."""
  return ldl_raw(a, lower=True, hermitian=False)


@jax.jit
def _sytrf_herm(a):
  """Raw hetrf/sytrf call only — complex Hermitian."""
  return ldl_raw(a, lower=True, hermitian=True)


@jax.jit
def _ldl_sym(a):
  """LDL factorisation (sytrf + BK untangle) — real symmetric."""
  return ldl_primitive(a, lower=True, hermitian=False)


@jax.jit
def _ldl_herm(a):
  """LDL factorisation (sytrf + BK untangle) — complex Hermitian."""
  return ldl_primitive(a, lower=True, hermitian=True)


@jax.jit
def _lu_solve_real(a, b):
  """LU full solve — real."""
  return jax.scipy.linalg.solve(a, b)


@jax.jit
def _lu_solve_cplx(a, b):
  """LU full solve — complex."""
  return jax.scipy.linalg.solve(a, b)


# ---------------------------------------------------------------------------
# LDL full solve wrappers — JIT over the JOINT factorization+solve graph
# so both ops are fused into a single XLA computation with no Python
# roundtrip between them.
# ---------------------------------------------------------------------------


@jax.jit
def _ldl_full_sym(a, b):
  """LDL full solve — real symmetric (factorization + solve, single JIT)."""
  factors, ipiv, perm = ldl_primitive(a, lower=True, hermitian=False)
  return ldl_solve(factors, ipiv, perm, b, lower=True, hermitian=False)


@jax.jit
def _ldl_full_herm(a, b):
  """LDL full solve — complex Hermitian (factorization + solve, single JIT)."""
  factors, ipiv, perm = ldl_primitive(a, lower=True, hermitian=True)
  return ldl_solve(factors, ipiv, perm, b, lower=True, hermitian=True)


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
  sytrf_fn,
  ldl_factor_fn,
  ldl_full_fn,
  lu_solve_fn,
) -> list[dict]:
  rows = []
  print(
    f"  {'n':>6}  {'sytrf (ms)':>10}  {'factor (ms)':>11}  "
    f"{'full (ms)':>10}  {'LU (ms)':>8}  {'ldl_vs_lu':>9}"
  )
  print("  " + "-" * 75)
  for n in SIZES:
    a = matrix_fn(n, dtype)
    b = jnp.ones(n, dtype=dtype)

    t_sytrf = time_jax_fn(sytrf_fn, a)
    t_ldl_fact = time_jax_fn(ldl_factor_fn, a)
    t_ldl_full = time_jax_fn(ldl_full_fn, a, b)
    t_lu_full = time_jax_fn(lu_solve_fn, a, b)

    speedup = t_lu_full / t_ldl_full if t_ldl_full > 0 else float("nan")
    print(
      f"  {n:>6d}  {t_sytrf * 1e3:10.2f}  {t_ldl_fact * 1e3:11.2f}  "
      f"{t_ldl_full * 1e3:10.2f}  {t_lu_full * 1e3:8.2f}  {speedup:9.2f}x"
    )

    rows.append({
      "series": label,
      "n": n,
      "sytrf_ms": t_sytrf * 1e3,
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
    _sytrf_sym,
    _ldl_sym,
    _ldl_full_sym,
    _lu_solve_real,
  )

  print("\n=== Complex Hermitian indefinite (c128) ===")
  all_rows += _run_series(
    "herm_c128",
    np.complex128,
    _herm_indef,
    _sytrf_herm,
    _ldl_herm,
    _ldl_full_herm,
    _lu_solve_cplx,
  )

  # Write CSV.
  csv_path = RESULTS_DIR / "bench_ldl.csv"
  fieldnames = [
    "series",
    "n",
    "sytrf_ms",
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
      [r["sytrf_ms"] for r in data],
      "^:",
      label="sytrf only (raw LAPACK)",
      linewidth=1.5,
      markersize=4,
      alpha=0.7,
    )
    ax.loglog(
      ns,
      [r["ldl_factoriz_ms"] for r in data],
      "o-",
      label="ldl factor (sytrf + untangle)",
      linewidth=2,
      markersize=5,
    )
    ax.loglog(
      ns,
      [r["ldl_solve_ms"] for r in data],
      "s--",
      label="ldl full solve",
      linewidth=2,
      markersize=5,
    )
    ax.loglog(
      ns,
      [r["lu_solve_ms"] for r in data],
      "D:",
      label="LU full solve (scipy)",
      linewidth=1.5,
      markersize=4,
      alpha=0.7,
    )

    ax.set_xlabel("Matrix size n", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

  fig.suptitle(
    "LDL benchmark: sytrf only vs factor (w/ untangle) vs full solve", fontsize=11
  )
  fig.tight_layout()
  png_path = RESULTS_DIR / "bench_ldl.png"
  fig.savefig(png_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Plot saved to {png_path}")


if __name__ == "__main__":
  main()
