"""Benchmark: LDL vs LU vs Cholesky for symmetric/Hermitian systems.

Compares wall-clock time for:
  - LDL (sytrf) via jaxtra  — for symmetric/Hermitian indefinite matrices
  - LU  (getrf) via jax.lax.linalg.lu  — always applicable
  - Cholesky via jax.lax.linalg.cholesky  — only for PSD matrices

Matrix sizes: 50 to 10 000.  Two series:
  1. Real symmetric indefinite (f64)
  2. Complex Hermitian indefinite (c128)

Results are written to benchmarks/results/bench_ldl.csv and companion PNGs.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax._src.lax.linalg import lu as jax_lu, cholesky as jax_cholesky

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import ldl as ldl_primitive

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SIZES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
REPEATS = 5
RNG = np.random.default_rng(42)


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
# Timing helper
# ---------------------------------------------------------------------------


def _bench(fn, *args, repeats=REPEATS) -> float:
  """Return minimum wall-clock time (seconds) over `repeats` runs."""
  # Warm up.
  result = fn(*args)
  jax.block_until_ready(result)
  times = []
  for _ in range(repeats):
    t0 = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    times.append(time.perf_counter() - t0)
  return min(times)


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _run_series(label: str, dtype, matrix_fn, hermitian: bool) -> list[dict]:
  rows = []
  for n in SIZES:
    a = matrix_fn(n, dtype)
    a_jit = jax.jit(lambda x: ldl_primitive(x, lower=True, hermitian=hermitian))
    lu_jit = jax.jit(jax_lu)

    t_ldl = _bench(a_jit, a)
    t_lu = _bench(lu_jit, a)

    row = {
      "series": label,
      "n": n,
      "ldl_s": t_ldl,
      "lu_s": t_lu,
      "ldl_vs_lu_speedup": t_lu / t_ldl if t_ldl > 0 else float("nan"),
    }

    # Cholesky is only valid for PSD; skip for indefinite matrices but include
    # a row entry for comparison when a PSD matrix is used.
    if not hermitian:
      a_psd = _sym_psd(n, dtype)
      chol_jit = jax.jit(jax_cholesky)
      t_chol = _bench(chol_jit, a_psd)
      row["cholesky_s"] = t_chol
    else:
      row["cholesky_s"] = float("nan")

    rows.append(row)
    print(
      f"  {label}  n={n:5d}: LDL={t_ldl * 1e3:7.2f} ms  "
      f"LU={t_lu * 1e3:7.2f} ms  "
      f"speedup={row['ldl_vs_lu_speedup']:.2f}x"
    )
  return rows


def main():
  all_rows: list[dict] = []

  print("=== Real symmetric indefinite (f64) ===")
  all_rows += _run_series("sym_f64", np.float64, _sym_indef, hermitian=False)

  print("=== Complex Hermitian indefinite (c128) ===")
  all_rows += _run_series(
    "herm_c128", np.complex128, _herm_indef, hermitian=True
  )

  # Write CSV.
  csv_path = RESULTS_DIR / "bench_ldl.csv"
  fieldnames = [
    "series",
    "n",
    "ldl_s",
    "lu_s",
    "cholesky_s",
    "ldl_vs_lu_speedup",
  ]
  with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
  print(f"\nResults written to {csv_path}")

  # Optional: generate PNG plots if matplotlib is available.
  try:
    _plot(all_rows)
  except ImportError:
    print("matplotlib not available — skipping plots")


def _plot(rows: list[dict]):
  import matplotlib.pyplot as plt

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  for ax, series in zip(axes, ["sym_f64", "herm_c128"]):
    data = [r for r in rows if r["series"] == series]
    ns = [r["n"] for r in data]
    ax.loglog(ns, [r["ldl_s"] * 1e3 for r in data], "o-", label="LDL (sytrf)")
    ax.loglog(ns, [r["lu_s"] * 1e3 for r in data], "s--", label="LU (getrf)")
    if series == "sym_f64":
      chol = [r["cholesky_s"] * 1e3 for r in data]
      if not all(np.isnan(chol)):
        ax.loglog(ns, chol, "^:", label="Cholesky (PSD only)")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Time (ms)")
    ax.set_title(series)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

  fig.tight_layout()
  png_path = RESULTS_DIR / "bench_ldl.png"
  fig.savefig(png_path, dpi=150)
  print(f"Plot saved to {png_path}")


if __name__ == "__main__":
  main()
