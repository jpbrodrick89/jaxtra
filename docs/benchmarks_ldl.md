# LDL Benchmarks

Comparison of jaxtra's LDL (Bunch-Kaufman) factorisation against LU for a
full solve (factorization + triangular solve) of a single 1-D right-hand side,
on square matrices (float64 and complex128, CPU, n = 50 – 5000):

| Method            | Implementation                                                               |
| ----------------- | ---------------------------------------------------------------------------- |
| **LDL full solve**| `jaxtra._src.lax.linalg.ldl` (JIT) + `ldl_solve` via LAPACK `sytrs` (JIT)  |
| **LU full solve** | `jax.scipy.linalg.solve` — fully JIT-compiled LAPACK `dgetrf`/`dtrsv`       |

All timings use `jax.block_until_ready` with one warmup run followed by five
timed repetitions; the reported value is the median.
Raw results are in `benchmarks/results/bench_ldl.csv`.

**Note on D storage**: D is returned as a full `(n, n)` block-diagonal matrix,
not a 1-D array. Bunch-Kaufman pivoting can produce 2×2 off-diagonal blocks in
D, so a 1-D diagonal representation is insufficient. This matches scipy's
behavior.

## Results

```{figure} ../benchmarks/results/bench_ldl.png
:alt: LDL vs LU full solve benchmark
:width: 95%
:align: center
```

## Summary

**LDL factorization vs LU factorization**
: LDL factorization alone is competitive with LU and faster for large symmetric
problems (1.2 – 1.6× speedup for complex Hermitian at n ≥ 500).

**LDL full solve vs LU full solve**
: Both `ldl_solve` (via LAPACK `sytrs`/`hetrs`) and `jax.scipy.linalg.solve`
are fully JIT-compiled, making timings directly comparable. LDL is faster than
LU for large symmetric indefinite problems (n ≥ 2000 for f64; n ≥ 500 for
c128), with speedups of 1.1 – 1.4× at n = 5000. For small n, LU's lower
factorization overhead wins, but LDL is within 2× across all sizes.

**Qualitative take-away**: `jaxtra.scipy.linalg.ldl` + `ldl_solve` is the
right choice for large symmetric/Hermitian indefinite systems, or when you need
the factorization itself (e.g. to extract D or the permutation for downstream
use), or when batching via `vmap`. For small matrices or when symmetry cannot
be exploited, `jax.scipy.linalg.solve` remains a fine default.
