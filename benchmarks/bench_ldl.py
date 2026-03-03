"""Benchmark: LDL vs LU vs Cholesky for symmetric/Hermitian systems.

Compares wall-clock time for:
  - LDL (sytrf) via jaxtra  — for symmetric/Hermitian indefinite matrices
  - LU  (getrf) via jax.lax.linalg.lu  — always applicable
  - Cholesky via jax.lax.linalg.cholesky  — only for PSD matrices

Matrix sizes: 50 to 10 000.  Two series:
  1. Real symmetric indefinite (f64)
  2. Complex Hermitian indefinite (c128)

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
from jax._src.lax.linalg import lu as jax_lu, cholesky as jax_cholesky

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import ldl as ldl_primitive

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
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
  """LU factorisation."""
  return jax_lu(a)


@jax.jit
def _cholesky(a):
  """Cholesky factorisation (PSD only)."""
  return jax_cholesky(a)


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


def _sym_psd(n: int, dtype) -> jax.Array:
  """Real symmetric positive-definite matrix (for Cholesky baseline)."""
  a = RNG.standard_normal((n, n)).astype(dtype)
  return jnp.array(a @ a.T + n * np.eye(n, dtype=dtype))


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _run_series(
  label: str,
  dtype,
  matrix_fn,
  ldl_fn,
  hermitian: bool,
) -> list[dict]:
  rows = []
  print(
    f"  {'n':>6}  {'LDL (ms)':>10}  {'LU (ms)':>10}  "
    f"{'Cholesky (ms)':>14}  {'LDL vs LU':>10}"
  )
  print("  " + "-" * 58)
  for n in SIZES:
    a = matrix_fn(n, dtype)

    t_ldl = time_jax_fn(ldl_fn, a)
    t_lu = time_jax_fn(_lu, a)

    if not hermitian:
      a_psd = _sym_psd(n, dtype)
      t_chol = time_jax_fn(_cholesky, a_psd)
      chol_str = f"{t_chol * 1e3:14.2f}"
    else:
      t_chol = float("nan")
      chol_str = f"{'—':>14}"

    speedup = t_lu / t_ldl if t_ldl > 0 else float("nan")
    print(
      f"  {n:>6d}  {t_ldl * 1e3:10.2f}  {t_lu * 1e3:10.2f}  "
      f"{chol_str}  {speedup:9.2f}x"
    )

    rows.append({
      "series": label,
      "n": n,
      "ldl_ms": t_ldl * 1e3,
      "lu_ms": t_lu * 1e3,
      "cholesky_ms": t_chol * 1e3,
      "ldl_vs_lu_speedup": speedup,
    })
  return rows


def main():
  all_rows: list[dict] = []

  print("=== Real symmetric indefinite (f64) ===")
  all_rows += _run_series(
    "sym_f64", np.float64, _sym_indef, _ldl_sym, hermitian=False
  )

  print("\n=== Complex Hermitian indefinite (c128) ===")
  all_rows += _run_series(
    "herm_c128", np.complex128, _herm_indef, _ldl_herm, hermitian=True
  )

  # Write CSV.
  csv_path = RESULTS_DIR / "bench_ldl.csv"
  fieldnames = [
    "series",
    "n",
    "ldl_ms",
    "lu_ms",
    "cholesky_ms",
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
      [r["ldl_ms"] for r in data],
      "o-",
      label="LDL (sytrf)",
      linewidth=2,
      markersize=5,
    )
    ax.loglog(
      ns,
      [r["lu_ms"] for r in data],
      "s--",
      label="LU (getrf)",
      linewidth=2,
      markersize=5,
    )
    if series == "sym_f64":
      chol = [r["cholesky_ms"] for r in data]
      if not all(np.isnan(chol)):
        ax.loglog(
          ns,
          chol,
          "^:",
          label="Cholesky (PSD only)",
          linewidth=2,
          markersize=5,
        )

    ax.set_xlabel("Matrix size n", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

  fig.tight_layout()
  png_path = RESULTS_DIR / "bench_ldl.png"
  fig.savefig(png_path, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Plot saved to {png_path}")


if __name__ == "__main__":
  main()
