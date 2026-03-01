"""
Benchmark: jaxtra ORMQR vs dense QR vs scipy lstsq (SVD)
=========================================================
For an overdetermined system A @ x ≈ b:

  jaxtra : geqrf(A) + ormqr(H, taus, b) + solve_triangular  (Q never formed)
  dense QR: jnp.linalg.qr(A) + Q.T @ b + solve_triangular   (Q materialized)
  scipy   : scipy.linalg.lstsq  (SVD-based, NumPy arrays)

Expected speedups (from PR observations):
  ~1.7× over dense QR
  ~2–3.5× over scipy lstsq (SVD)

Usage:
  python benchmarks/bench_least_squares.py
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from scipy import linalg as scipy_linalg

from jaxtra.scipy.linalg import qr_multiply

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


def _scipy_svd_solve(A_np, b_np):
    """Least-squares via scipy.linalg.lstsq (SVD)."""
    x, _, _, _ = scipy_linalg.lstsq(A_np, b_np)
    return x


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_jax_fn(fn, *args, n_warmup=1, n_repeat=5):
    """Run `fn(*args)` with block_until_ready, return median wall-time (s)."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_numpy_fn(fn, *args, n_warmup=1, n_repeat=5):
    """Time a plain NumPy/SciPy function."""
    for _ in range(n_warmup):
        fn(*args)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SIZES = [
    (500,   20),
    (2000,  50),
    (5000, 100),
    (10000, 100),
]

N_WARMUP = 1
N_REPEAT = 5
RNG = np.random.default_rng(0)

print(f"{'Shape':>14}  {'jaxtra (ms)':>12}  {'dense QR (ms)':>14}  "
      f"{'scipy SVD (ms)':>15}  {'vs QR':>6}  {'vs SVD':>7}")
print("-" * 80)

for m, n in SIZES:
    # Generate data (float64 to match scipy defaults)
    A_np = RNG.standard_normal((m, n)).astype(np.float64)
    b_np = RNG.standard_normal(m).astype(np.float64)
    A_jax = jnp.array(A_np)
    b_jax = jnp.array(b_np)

    t_jaxtra  = time_jax_fn(_jaxtra_solve,   A_jax, b_jax,
                             n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_dense   = time_jax_fn(_dense_qr_solve, A_jax, b_jax,
                             n_warmup=N_WARMUP, n_repeat=N_REPEAT)
    t_scipy   = time_numpy_fn(_scipy_svd_solve, A_np, b_np,
                               n_warmup=N_WARMUP, n_repeat=N_REPEAT)

    speedup_qr  = t_dense  / t_jaxtra
    speedup_svd = t_scipy  / t_jaxtra

    print(f"({m:5d},{n:4d})  "
          f"{t_jaxtra*1e3:12.2f}  "
          f"{t_dense*1e3:14.2f}  "
          f"{t_scipy*1e3:15.2f}  "
          f"{speedup_qr:6.2f}x  "
          f"{speedup_svd:6.2f}x")

print()
print("Speedup = time(reference) / time(jaxtra)  — higher is better for jaxtra")
