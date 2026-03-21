"""
Benchmark: jaxtra ormqr vs householder_product vs parallel T-reduction
======================================================================
Compares Q-multiply from (R, taus) — three competing strategies:

  ormqr              : geqrf(A) → ormqr(R, taus, c)           (Q never formed, LAPACK/cuSOLVER)
  householder_product: geqrf(A) → householder_product → Q.T@c (Q materialised, orgqr)
  parallel T-reduce  : geqrf(A) → SYRK + log(k) tree → YTY^T (pure JAX, no Q formed)

All start from a pre-computed geqrf factorisation so only the
Q-multiply step is timed (geqrf cost is excluded).

Results are written to  benchmarks/results/bench_ormqr[_gpu].csv
Plots are written to    benchmarks/results/bench_ormqr_cols{N}[_gpu].png

Usage:
  python benchmarks/bench_ormqr.py           # CPU run
  python benchmarks/bench_ormqr.py --gpu     # GPU run with larger sizes
"""

import jax
jax.config.update("jax_enable_x64", True)

import argparse
import csv
import pathlib
import time
import timeit

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as jlax
from jax import vmap
from jax._src.lax import lax
from jax._src.lax.linalg import geqrf, householder_product

from jaxtra._src.lax.linalg import ormqr

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true",
                    help="GPU mode: also benchmark at 50k and 100k rows, "
                         "append '_gpu' to output filenames.")
args = parser.parse_args()

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUFFIX = "_gpu" if args.gpu else ""

# ---------------------------------------------------------------------------
# JIT-compiled Q-multiply paths (geqrf already done, only multiply timed)
# ---------------------------------------------------------------------------

@jax.jit
def _ormqr_multiply(r, taus, c):
    """ormqr: apply Householder reflectors directly — Q never formed."""
    return ormqr(r, taus, c, left=True, transpose=True)


@jax.jit
def _householder_multiply(r, taus, c):
    """householder_product path: form Q explicitly, then matmul.

    Mirrors _qr_lowering's logic for building Q from (r, taus).
    """
    m, n = r.shape[-2:]
    if m < n:
        q = householder_product(r[..., :m, :m], taus)
    elif m > n:
        # Pad r to (m, m) exactly as _qr_lowering does with lax.pad.
        pads = [(0, 0, 0)] * (r.ndim - 1) + [(0, m - n, 0)]
        q = lax.pad(r, lax._zero(r), pads)
        q = householder_product(q, taus)
    else:
        q = householder_product(r, taus)
    return q.T @ c


# ---------------------------------------------------------------------------
# Parallel blocked Householder ormqr via associative T-reduction
# ---------------------------------------------------------------------------

def _extract_householder(a):
    """Extract Y from packed geqrf output: subdiag of a + unit diagonal."""
    m, k = a.shape[-2:]
    Y = jnp.tril(a, k=-1)
    Y = Y.at[jnp.arange(k), jnp.arange(k)].set(1.0)
    return Y  # (m, k)


def _combine_T(T_a, T_b, G_sub):
    """Combine two lower-triangular T blocks: T_a ⊕ T_b = [[T_a, 0], [S, T_b]]."""
    s = T_a.shape[0]
    S = -T_b @ G_sub @ T_a
    top = jnp.concatenate([T_a, jnp.zeros_like(S)], axis=1)
    bot = jnp.concatenate([S, T_b], axis=1)
    return jnp.concatenate([top, bot], axis=0)


def _apply_block_T(T, Y_block, C):
    """C ← (I - Y_block @ T @ Y_block^T) @ C — three GEMMs."""
    W = T @ (Y_block.T @ C)
    return C - Y_block @ W


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _parallel_ormqr(a, taus, C, transpose=False):
    """Apply Q (or Q^T) via parallel T-reduction.

    Args:
        a:         (m, k) packed geqrf output
        taus:      (k,) Householder scalars
        C:         (m, p) matrix to transform
        transpose: if True, compute Q^T @ C instead of Q @ C

    Returns:
        Q @ C or Q^T @ C, shape (m, p)
    """
    m, k = a.shape[-2:]

    Y = _extract_householder(a)

    # Pad k to next power of 2 with zero taus (identity reflectors)
    k_padded = _next_power_of_2(k)
    if k_padded > k:
        taus = jnp.concatenate([taus, jnp.zeros(k_padded - k, dtype=taus.dtype)])
        Y = jnp.concatenate([Y, jnp.zeros((m, k_padded - k), dtype=Y.dtype)], axis=1)

    # Single SYRK: G = Y^T Y
    # symmetric_product(A, C) computes A @ A^T, so pass Y^T to get Y^T @ Y
    G = jax.lax.linalg.symmetric_product(
        Y.T, jnp.zeros((k_padded, k_padded), dtype=Y.dtype),
        symmetrize_output=True)

    # Leaf T blocks: (k_padded, 1, 1), each a scalar tau_i
    T = taus[:, None, None]
    s = 1

    while T.shape[0] > 1:
        n_pairs = T.shape[0] // 2

        T_pairs = T.reshape(n_pairs, 2, s, s)
        T_a = T_pairs[:, 0]
        T_b = T_pairs[:, 1]

        pair_starts = jnp.arange(n_pairs) * (2 * s)

        def get_G_sub(i):
            return jlax.dynamic_slice(G, (i + s, i), (s, s))

        G_subs = vmap(get_G_sub)(pair_starts)
        T = vmap(_combine_T)(T_a, T_b, G_subs)
        s *= 2

    # Final application
    T_final = T[0]
    # The tree's lower-tri T gives Q^T (H_1 applied first).
    # For Q^T @ C use T directly; for Q @ C use T^T.
    T_apply = T_final if transpose else T_final.T
    C = _apply_block_T(T_apply, Y, C)
    return C


@jax.jit
def _parallel_T_multiply(r, taus, c):
    """Parallel T-reduction: Q^T @ c via SYRK + log(k) tree."""
    return _parallel_ormqr(r, taus, c, transpose=True)


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness():
    """Verify parallel T-reduction matches ormqr before benchmarking."""
    print("Correctness check...")
    rng = np.random.default_rng(42)
    for m, k in [(32, 8), (64, 16), (100, 50), (128, 128), (200, 63)]:
        A = jnp.array(rng.standard_normal((m, k)), dtype=jnp.float64)
        c = jnp.array(rng.standard_normal((m, 1)), dtype=jnp.float64)
        r, taus = geqrf(A)
        r = jax.block_until_ready(r)
        taus = jax.block_until_ready(taus)

        ref = jax.block_until_ready(_ormqr_multiply(r, taus, c))
        got = jax.block_until_ready(_parallel_T_multiply(r, taus, c))
        err = float(jnp.max(jnp.abs(ref - got)))
        status = "OK" if err < 1e-10 else "FAIL"
        print(f"  ({m:>4d}, {k:>3d})  max|diff| = {err:.2e}  {status}")
        if err >= 1e-10:
            raise AssertionError(
                f"parallel T-reduction failed for (m={m}, k={k}): "
                f"max|diff| = {err:.2e} (tol 1e-10)")
    print("  All passed.\n")


check_correctness()


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_jax_fn(fn, *args):
    """Warm up via JIT, then autorange to get per-call time in seconds."""
    jax.block_until_ready(fn(*args))
    timer = timeit.Timer(lambda: jax.block_until_ready(fn(*args)))
    n_iter, total = timer.autorange()
    return total / n_iter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COL_COUNTS = [20, 50, 100]
ROW_SIZES  = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
if args.gpu:
    COL_COUNTS = COL_COUNTS + [200, 512, 768]
    ROW_SIZES  = ROW_SIZES + [50_000, 100_000, 200_000]

RNG = np.random.default_rng(0)

METHODS = [
    ("ormqr",               "#1f77b4", "o"),
    ("parallel T-reduce",   "#d62728", "^"),
    ("householder_product", "#ff7f0e", "s"),
]

# ---------------------------------------------------------------------------
# Run benchmarks
# ---------------------------------------------------------------------------

records = []

for n_cols in COL_COUNTS:
    print(f"\nn_cols = {n_cols}")
    print(f"  {'n_rows':>7}  {'ormqr (ms)':>12}  {'parallel (ms)':>14}  "
          f"{'householder (ms)':>17}  {'par/ormqr':>10}  {'hp/ormqr':>9}")
    print("  " + "-" * 75)

    for n_rows in ROW_SIZES:
        if n_rows < n_cols:
            continue

        # Generate data and pre-compute geqrf
        A_np = RNG.standard_normal((n_rows, n_cols)).astype(np.float64)
        c_np = RNG.standard_normal((n_rows, 1)).astype(np.float64)
        A_jx = jax.block_until_ready(jax.device_put(A_np))
        c_jx = jax.block_until_ready(jax.device_put(c_np))

        r, taus = geqrf(A_jx)
        r = jax.block_until_ready(r)
        taus = jax.block_until_ready(taus)

        t_ormqr = time_jax_fn(_ormqr_multiply, r, taus, c_jx)
        t_par   = time_jax_fn(_parallel_T_multiply, r, taus, c_jx)
        t_hp    = time_jax_fn(_householder_multiply, r, taus, c_jx)

        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "ormqr", "time_ms": t_ormqr * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "parallel T-reduce", "time_ms": t_par * 1e3})
        records.append({"n_rows": n_rows, "n_cols": n_cols,
                         "method": "householder_product", "time_ms": t_hp * 1e3})

        print(f"  {n_rows:>7d}  {t_ormqr*1e3:12.2f}  {t_par*1e3:14.2f}  "
              f"{t_hp*1e3:17.2f}  {t_par/t_ormqr:9.2f}x  {t_hp/t_ormqr:8.2f}x")

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / f"bench_ormqr{SUFFIX}.csv"
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
    ax.set_title(f"Q-multiply from (R, taus)  —  {n_cols} columns", fontsize=13)
    ax.set_xticks(ROW_SIZES)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    out = RESULTS_DIR / f"bench_ormqr_cols{n_cols}{SUFFIX}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out}")

print("\nSpeedup columns: par/ormqr and hp/ormqr — lower is better for the challenger")
