# LDL Benchmarks

Comparison of jaxtra's LDL (Bunch-Kaufman) factorisation against LU for a
full solve (factorization + triangular solve) of a single 1-D right-hand side,
on square matrices (float64 and complex128, CPU, n = 50 – 5000):

| Method             | Implementation                                                              |
| ------------------ | --------------------------------------------------------------------------- |
| **LDL full solve** | `jaxtra._src.lax.linalg.ldl` (JIT) + `ldl_solve` via LAPACK `sytrs` (JIT) |
| **LU full solve**  | `jax.scipy.linalg.solve` — fully JIT-compiled LAPACK `dgetrf`/`dtrsv`      |

Both the factorization and the triangular solve are fused into a single XLA
computation (one `@jax.jit` scope), so there is no Python round-trip overhead
between them.  All timings use `jax.block_until_ready` with one warmup run
followed by five timed repetitions; the reported value is the median.
Raw results are in `benchmarks/results/bench_ldl.csv`.

**Note on D storage**: D is returned as a full `(n, n)` block-diagonal matrix,
not a 1-D array.  Bunch-Kaufman pivoting can produce 2×2 off-diagonal blocks in
D, so a 1-D diagonal representation is insufficient.  This matches scipy's
behaviour.

## Results (Linux / OpenBLAS, GitHub Actions)

```{figure} ../benchmarks/results/bench_ldl.png
:alt: LDL vs LU full solve benchmark
:width: 95%
:align: center
```

### Real symmetric indefinite (f64)

| n    | LDL factorization (ms) | LDL full solve (ms) | LU full solve (ms) | LDL / LU |
| ---- | ---------------------- | ------------------- | ------------------ | --------- |
| 50   | 0.07                   | 0.04                | 0.09               | **2.27×** |
| 100  | 0.09                   | 0.08                | 0.13               | **1.52×** |
| 200  | 0.38                   | 0.35                | 0.67               | **1.91×** |
| 500  | 4.98                   | 3.06                | 2.98               | ~1×       |
| 1000 | 17.60                  | 14.19               | 14.53              | ~1×       |
| 2000 | 105.4                  | 74.9                | 122.7              | **1.64×** |
| 5000 | 1011                   | 917                 | 1236               | **1.35×** |

### Complex Hermitian indefinite (c128)

| n    | LDL factorization (ms) | LDL full solve (ms) | LU full solve (ms) | LDL / LU |
| ---- | ---------------------- | ------------------- | ------------------ | --------- |
| 50   | 0.06                   | 0.07                | 0.09               | **1.41×** |
| 100  | 0.12                   | 0.13                | 0.27               | **2.16×** |
| 200  | 0.76                   | 0.78                | 1.07               | **1.37×** |
| 500  | 5.40                   | 5.24                | 7.19               | **1.37×** |
| 1000 | 31.3                   | 29.1                | 40.5               | **1.39×** |
| 2000 | 297                    | 233                 | 353                | **1.51×** |
| 5000 | 2779                   | 2561                | 4114               | **1.61×** |

## Summary

**Complex Hermitian (c128)**: LDL is faster than LU at every tested size
(1.37–2.16×), exploiting Hermitian structure via LAPACK `hetrf`/`hetrs`.

**Real symmetric (f64)**: LDL is faster at small (n ≤ 200) and large (n ≥ 2000)
sizes, roughly equal at n = 500–1000, consistent with n³/6 arithmetic advantage
of symmetric factorization becoming more pronounced at large n.

### Platform note — macOS vs Linux

On macOS (Accelerate framework), `getrf` is more aggressively hand-tuned than
`sytrf`, so LU can be faster for real symmetric solves at medium sizes.  On
Linux (OpenBLAS, as measured here on GitHub Actions), `getrf` is not as
optimised relative to `sytrf`, so LDL shows broader speedups.  The speedup on
GHA is mainly driven by **`getrf` being slower on OpenBLAS** rather than
`sytrf` being dramatically faster — `sytrf` at n = 2000 is ~105 ms on GHA vs
~47 ms on Mac (2.2× difference), while `getrf` at n = 2000 is ~123 ms on GHA
vs ~22 ms on Mac (5.6× difference).

> **Warning**: on some platforms (e.g. macOS with Accelerate), `sytrf` may be
> slower than `getrf` for purely real symmetric matrices.  Benchmark both paths
> for your hardware before optimising.

### Qualitative take-away

`jaxtra.scipy.linalg.ldl` + `ldl_solve` is the right choice for
symmetric/Hermitian indefinite systems when you:

- are on Linux (OpenBLAS/MKL), or
- have complex Hermitian matrices (consistent speedup everywhere), or
- need the factorization itself (to extract D or the permutation), or
- are batching via `vmap`.

For small real symmetric matrices on macOS, `jax.scipy.linalg.solve` may still
be faster — benchmark to confirm.

## Reproduction

```bash
uv run --locked python benchmarks/bench_ldl.py
```
