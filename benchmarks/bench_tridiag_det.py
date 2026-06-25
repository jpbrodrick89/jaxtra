"""
Benchmark: tridiagonal log-det — four implementations
======================================================

  simple : raw three-term recurrence, unroll=16  (overflows for large n)
  stable : signed log-space recurrence, unroll=1
  scan   : signed log-space recurrence, unroll=8, indexes via jnp.arange
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
    """O(n) log-det via raw recurrence. Overflows for n > ~1500 at typical scales."""
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, xs):
        f_prev2, f_prev1 = carry
        b_i, coupling_i = xs
        return (f_prev1, b_i * f_prev1 - coupling_i * f_prev2), None

    (_, f_n), _ = lax.scan(step, (1.0, b[0]), (b[1:], coupling[1:]), unroll=16)
    return jnp.sign(f_n), jnp.log(jnp.abs(f_n))


@jax.jit
def tridiag_logdet_stable(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """O(n) log-det via signed log-space recurrence. Overflow-safe for any n."""
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, xs):
        log_f2, sgn_f2, log_f1, sgn_f1 = carry
        b_i, coupling_i = xs
        log_t1  = jnp.log(jnp.abs(b_i))        + log_f1
        sgn_t1  = jnp.sign(b_i)               * sgn_f1
        log_t2  = jnp.log(jnp.abs(coupling_i)) + log_f2
        sgn_t2  = jnp.sign(coupling_i)         * sgn_f2
        log_max = jnp.maximum(log_t1, log_t2)
        val     = sgn_t1 * jnp.exp(log_t1 - log_max) - sgn_t2 * jnp.exp(log_t2 - log_max)
        return (log_f1, sgn_f1, log_max + jnp.log(jnp.abs(val)), jnp.sign(val)), None

    init = (0.0, 1.0, jnp.log(jnp.abs(b[0])), jnp.sign(b[0]))
    (_, _, log_det, sgn_det), _ = lax.scan(step, init, (b[1:], coupling[1:]), unroll=1)
    return sgn_det, log_det


@jax.jit
def tridiag_logdet_scan(
    a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Signed log-space recurrence, unroll=8, indexes into b/coupling at runtime."""
    coupling = jnp.concatenate([jnp.array([0.0]), a * c])

    def step(carry, i):
        log_f2, sgn_f2, log_f1, sgn_f1 = carry
        log_t1  = jnp.log(jnp.abs(b[i]))        + log_f1
        sgn_t1  = jnp.sign(b[i])               * sgn_f1
        log_t2  = jnp.log(jnp.abs(coupling[i])) + log_f2
        sgn_t2  = jnp.sign(coupling[i])         * sgn_f2
        log_max = jnp.maximum(log_t1, log_t2)
        val     = sgn_t1 * jnp.exp(log_t1 - log_max) - sgn_t2 * jnp.exp(log_t2 - log_max)
        return (log_f1, sgn_f1, log_max + jnp.log(jnp.abs(val)), jnp.sign(val)), None

    init = (0.0, 1.0, jnp.log(jnp.abs(b[0])), jnp.sign(b[0]))
    (_, _, log_det, sgn_det), _ = lax.scan(
        step, init, jnp.arange(1, b.shape[0]), unroll=8)
    return sgn_det, log_det


# Batched variants
_simple_b = jax.jit(jax.vmap(tridiag_logdet_simple))
_stable_b = jax.jit(jax.vmap(tridiag_logdet_stable))
_scan_b   = jax.jit(jax.vmap(tridiag_logdet_scan))
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

def check_correctness(n=256):
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
        "stable": tridiag_logdet_stable(a, b, c),
        "scan":   tridiag_logdet_scan(a, b, c),
        "gttrf":  tridiag_logdet_lu(a, b, c),
    }
    for name, (_, ldet) in results.items():
        err = abs(float(ldet) - ref)
        print(f"  [{name:6s}] logdet={float(ldet):.6f}  err={err:.2e}  "
              f"{'OK' if err < 1e-9 else 'FAIL'}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIZES = [64, 128, 256, 512, 1024, 2048, 4096]
BATCH = 64
RNG   = np.random.default_rng(42)

IMPLS = [
    # (key, label, color, marker)
    ("simple", "simple (unroll=16)", "#2ca02c", "^"),
    ("stable", "stable (unroll=1)",  "#d62728", "v"),
    ("scan",   "scan   (unroll=8)",  "#1f77b4", "o"),
    ("gttrf",  "gttrf (LAPACK)",     "#ff7f0e", "s"),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print(f"Correctness check (n=256):")
check_correctness(256)

hdr = (f"\n{'n':>5}  "
       + "  ".join(f"{k+' (ms)':>13}" for k, *_ in IMPLS)
       + f"  {'vs simple':>10}  {'vs stable':>10}  {'vs scan':>9}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

records = []

for n in SIZES:
    a  = jnp.array(RNG.standard_normal(n - 1) * 0.5)
    b  = jnp.array(RNG.standard_normal(n) + 4.0)
    c  = jnp.array(RNG.standard_normal(n - 1) * 0.5)
    A_b = jnp.array(RNG.standard_normal((BATCH, n - 1)) * 0.5)
    B_b = jnp.array(RNG.standard_normal((BATCH, n)) + 4.0)
    C_b = jnp.array(RNG.standard_normal((BATCH, n - 1)) * 0.5)

    fns_ub = {
        "simple": (jax.jit(tridiag_logdet_simple), (a, b, c)),
        "stable": (jax.jit(tridiag_logdet_stable), (a, b, c)),
        "scan":   (jax.jit(tridiag_logdet_scan),   (a, b, c)),
        "gttrf":  (jax.jit(tridiag_logdet_lu),     (a, b, c)),
    }
    fns_b = {
        "simple": (_simple_b, (A_b, B_b, C_b)),
        "stable": (_stable_b, (A_b, B_b, C_b)),
        "scan":   (_scan_b,   (A_b, B_b, C_b)),
        "gttrf":  (_gttrf_b,  (A_b, B_b, C_b)),
    }

    times_ub = {k: time_jax_fn(fn, *args) for k, (fn, args) in fns_ub.items()}
    times_b  = {k: time_jax_fn(fn, *args) for k, (fn, args) in fns_b.items()}

    g_ub = times_ub["gttrf"]
    g_b  = times_b["gttrf"]

    row = f"{n:>5d}  " + "  ".join(f"{times_ub[k]*1e3:13.3f}" for k, *_ in IMPLS)
    row += (f"  {times_ub['simple']/g_ub:9.2f}x"
            f"  {times_ub['stable']/g_ub:9.2f}x"
            f"  {times_ub['scan']/g_ub:8.2f}x")
    print(row)

    for k, label, *_ in IMPLS:
        records.append({"n": n, "variant": "unbatched", "method": label,
                         "time_ms": times_ub[k] * 1e3})
        records.append({"n": n, "variant": f"batched={BATCH}", "method": label,
                         "time_ms": times_b[k]  * 1e3})

# Batched summary
print(f"\nBatched (batch={BATCH}):")
hdr_b = (f"{'n':>5}  "
         + "  ".join(f"{k+' (ms)':>13}" for k, *_ in IMPLS)
         + f"  {'vs simple':>10}  {'vs stable':>10}  {'vs scan':>9}")
print(hdr_b)
print("  " + "-" * (len(hdr_b) - 2))

RNG2 = np.random.default_rng(42)  # same seed → same data as above
for n in SIZES:
    row_data = {r["method"]: r["time_ms"]
                for r in records
                if r["n"] == n and r["variant"] == f"batched={BATCH}"}
    keys_to_label = {label: k for k, label, *_ in IMPLS}
    times = {k: row_data[label] for k, label, *_ in IMPLS}
    g = times["gttrf"]
    row = f"{n:>5d}  " + "  ".join(f"{times[k]:13.3f}" for k, *_ in IMPLS)
    row += (f"  {times['simple']/g:9.2f}x"
            f"  {times['stable']/g:9.2f}x"
            f"  {times['scan']/g:8.2f}x")
    print(row)

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
# Plot: 2 panels (unbatched / batched), all 4 methods
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

fig.suptitle("Tridiagonal log-det: four implementations", fontsize=14)
fig.tight_layout()
out = RESULTS_DIR / "bench_tridiag_det.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Plot  → {out}")
print("\n(speedup columns show time(X) / time(gttrf); >1 means gttrf is faster)")
