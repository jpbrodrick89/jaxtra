"""
Benchmark: tridiagonal log-det — five implementations
======================================================

  simple : raw three-term recurrence, unroll=16  (overflows for large n)
  frexp  : frexp/ldexp power-of-2 rescaling — stable, no per-step log/exp
  stable : signed log-space recurrence, unroll=1
  gttrf  : LAPACK gttrf tridiagonal LU → prod(U diagonal) + pivot sign

Both unbatched (single matrix) and batched (vmap, batch=64) are timed
across n = 64 … 4096.

Results → benchmarks/results/bench_tridiag_det.csv / .png
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
# Implementations
# ---------------------------------------------------------------------------

@jax.jit
def tridiag_logdet_simple(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Raw three-term recurrence, unroll=16.  Overflows for large n."""
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, xs):
        f2, f1 = carry
        b_i, coup_i = xs
        return (f1, b_i * f1 - coup_i * f2), None

    (_, f_n), _ = lax.scan(step, (1.0, b[0]), (b[1:], coupling[1:]), unroll=16)
    return jnp.sign(f_n), jnp.log(jnp.abs(f_n))


@jax.jit
def tridiag_logdet_frexp(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Power-of-2 rescaling via frexp/ldexp.

    Carry: (f2, f1, log2_sum) with f1, f2 ∈ (-1, 1) at all times.
    frexp extracts the IEEE 754 binary exponent as an integer (bit op),
    ldexp multiplies by a power of 2 (bit op).  No log/exp in the inner
    loop; one log at the very end.
    """
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, xs):
        f2, f1, log2_s = carry
        b_i, coup_i = xs

        f = b_i * f1 - coup_i * f2

        # Rescale f1 and f by the same 2^exp so both land in (-1, 1).
        # scale = max(|f1|, |f|) ensures neither overflows going into the
        # next step's multiply-add.
        _, exp = jnp.frexp(jnp.maximum(jnp.abs(f1), jnp.abs(f)))
        return (
            jnp.ldexp(f1, -exp),
            jnp.ldexp(f, -exp),
            log2_s + exp.astype(jnp.float64),
        ), None

    # Initial scaling: keep both the implicit D_{-1}=1 and D_0=b[0] in (-1,1).
    _, exp0 = jnp.frexp(jnp.maximum(jnp.abs(b[0]), 1.0))
    init = (
        jnp.ldexp(1.0, -exp0),
        jnp.ldexp(b[0], -exp0),
        exp0.astype(jnp.float64),
    )

    (_, f1_final, log2_s), _ = lax.scan(
        step, init, (b[1:], coupling[1:]), unroll=16)

    logabsdet = log2_s * jnp.log(jnp.array(2.0)) + jnp.log(jnp.abs(f1_final))
    return jnp.sign(f1_final), logabsdet


@jax.jit
def tridiag_logdet_stable(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Signed log-space recurrence, unroll=1.  Overflow-safe for any n."""
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, xs):
        log_f2, sgn_f2, log_f1, sgn_f1 = carry
        b_i, coup_i = xs
        log_t1  = jnp.log(jnp.abs(b_i))        + log_f1
        sgn_t1  = jnp.sign(b_i)               * sgn_f1
        log_t2  = jnp.log(jnp.abs(coup_i))     + log_f2
        sgn_t2  = jnp.sign(coup_i)             * sgn_f2
        log_max = jnp.maximum(log_t1, log_t2)
        val     = sgn_t1 * jnp.exp(log_t1 - log_max) - sgn_t2 * jnp.exp(log_t2 - log_max)
        return (log_f1, sgn_f1, log_max + jnp.log(jnp.abs(val)), jnp.sign(val)), None

    init = (0.0, 1.0, jnp.log(jnp.abs(b[0])), jnp.sign(b[0]))
    (_, _, log_det, sgn_det), _ = lax.scan(
        step, init, (b[1:], coupling[1:]), unroll=1)
    return sgn_det, log_det


# Batched variants
_simple_b = jax.jit(jax.vmap(tridiag_logdet_simple))
_frexp_b  = jax.jit(jax.vmap(tridiag_logdet_frexp))
_stable_b = jax.jit(jax.vmap(tridiag_logdet_stable))
_gttrf_b  = jax.jit(jax.vmap(tridiag_logdet_lu))

# ---------------------------------------------------------------------------
# Timing
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
# Correctness
# ---------------------------------------------------------------------------

def check_correctness(n=512):
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n) + 4.0
    a_np = rng.standard_normal(n - 1) * 0.5
    c_np = rng.standard_normal(n - 1) * 0.5
    A = np.diag(b_np) + np.diag(a_np, -1) + np.diag(c_np, 1)
    _, ref = jnp.linalg.slogdet(jnp.array(A))
    ref = float(ref)

    a, b, c = jnp.array(a_np), jnp.array(b_np), jnp.array(c_np)
    results = {
        "simple": tridiag_logdet_simple(a, b, c),
        "frexp":  tridiag_logdet_frexp(a, b, c),
        "stable": tridiag_logdet_stable(a, b, c),
        "gttrf":  tridiag_logdet_lu(a, b, c),
    }
    for name, (_, ldet) in results.items():
        err = abs(float(ldet) - ref)
        print(f"  [{name:6s}] logdet={float(ldet):.6f}  err={err:.2e}  "
              f"{'OK' if err < 1e-8 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
BATCH = 64
RNG   = np.random.default_rng(42)

IMPLS = [
    # (key, label, color, marker)
    ("simple", "simple (unroll=16)",    "#2ca02c", "^"),
    ("frexp",  "frexp  (unroll=16)",    "#9467bd", "D"),
    ("stable", "stable (unroll=1)",     "#d62728", "v"),
    ("gttrf",  "gttrf  (LAPACK)",       "#ff7f0e", "s"),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"Correctness check (n=512):")
check_correctness(512)

def _run_and_print(label, fns_ub, fns_b):
    hdr = (f"\n{label}\n"
           f"{'n':>5}  "
           + "  ".join(f"{k+' (ms)':>14}" for k, *_ in IMPLS)
           + f"  {'vs simple':>10}  {'vs stable':>10}")
    sep = "  " + "-" * (len(hdr.split('\n')[-1]) - 2)
    print(hdr)
    print(sep)
    rows = []
    for n in SIZES:
        times = {k: time_jax_fn(fn, *args) for k, (fn, args) in fns_ub[n].items()}
        g = times["gttrf"]
        row = f"{n:>5d}  " + "  ".join(f"{times[k]*1e3:14.3f}" for k, *_ in IMPLS)
        row += f"  {times['simple']/g:9.2f}x  {times['stable']/g:9.2f}x"
        print(row)
        rows.append((n, times))
    return rows

print()
records = []

# Pre-build all input arrays
inputs_ub = {}
inputs_b  = {}
for n in SIZES:
    a  = jnp.array(RNG.standard_normal(n - 1) * 0.5)
    b  = jnp.array(RNG.standard_normal(n) + 4.0)
    c  = jnp.array(RNG.standard_normal(n - 1) * 0.5)
    A_b = jnp.array(RNG.standard_normal((BATCH, n - 1)) * 0.5)
    B_b = jnp.array(RNG.standard_normal((BATCH, n)) + 4.0)
    C_b = jnp.array(RNG.standard_normal((BATCH, n - 1)) * 0.5)
    inputs_ub[n] = (a, b, c)
    inputs_b[n]  = (A_b, B_b, C_b)

fns_ub_by_n = {
    n: {
        "simple": (jax.jit(tridiag_logdet_simple), inputs_ub[n]),
        "frexp":  (jax.jit(tridiag_logdet_frexp),  inputs_ub[n]),
        "stable": (jax.jit(tridiag_logdet_stable),  inputs_ub[n]),
        "gttrf":  (jax.jit(tridiag_logdet_lu),      inputs_ub[n]),
    }
    for n in SIZES
}
fns_b_by_n = {
    n: {
        "simple": (_simple_b, inputs_b[n]),
        "frexp":  (_frexp_b,  inputs_b[n]),
        "stable": (_stable_b, inputs_b[n]),
        "gttrf":  (_gttrf_b,  inputs_b[n]),
    }
    for n in SIZES
}

rows_ub = _run_and_print(f"Unbatched (single matrix)", fns_ub_by_n, fns_b_by_n)
rows_b  = _run_and_print(f"Batched (batch={BATCH})",   fns_b_by_n,  fns_b_by_n)

for n, times in rows_ub:
    for k, label, *_ in IMPLS:
        records.append({"n": n, "variant": "unbatched", "method": label,
                         "time_ms": times[k] * 1e3})
for n, times in rows_b:
    for k, label, *_ in IMPLS:
        records.append({"n": n, "variant": f"batched={BATCH}", "method": label,
                         "time_ms": times[k] * 1e3})

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

csv_path = RESULTS_DIR / "bench_tridiag_det.csv"
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["n", "variant", "method", "time_ms"])
    writer.writeheader()
    writer.writerows(records)
print(f"\nResults → {csv_path}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
panel_variants = ["unbatched", f"batched={BATCH}"]
panel_titles   = ["Unbatched (single matrix)", f"Batched (batch={BATCH})"]

for ax, variant, title in zip(axes, panel_variants, panel_titles):
    for key, label, color, marker in IMPLS:
        pts = [(r["n"], r["time_ms"])
               for r in records
               if r["variant"] == variant and r["method"] == label]
        xs, ys = zip(*sorted(pts))
        ax.plot(xs, ys, label=label, color=color, marker=marker,
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
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

fig.suptitle("Tridiagonal log-det: five implementations", fontsize=14)
fig.tight_layout()
out = RESULTS_DIR / "bench_tridiag_det.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot  → {out}")
print("\n(speedup columns: time(X) / time(gttrf); >1 means gttrf is faster)")
