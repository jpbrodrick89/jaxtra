# LDL Benchmarks

Comparison of jaxtra's LDL (Bunch-Kaufman) factorisation against LU for a
full solve (factorization + triangular solve) of a single 1-D right-hand side,
on square matrices (float64 and complex128, CPU, n = 50 – 5000):

| Method            | Implementation                                                        |
| ----------------- | --------------------------------------------------------------------- |
| **LDL full solve**| `jaxtra._src.lax.linalg.ldl` (JIT) + `ldl_solve` (NumPy)            |
| **LU full solve** | `jax.scipy.linalg.solve` — fully JIT-compiled LAPACK `dgetrf`/`dtrsv`|

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
: `ldl_solve` is currently implemented in NumPy (not JIT-traceable), while
`jax.scipy.linalg.solve` is fully JIT-compiled. As a result the end-to-end
solve is significantly slower through the jaxtra path at all matrix sizes.
The unit-diagonal triangular factors do not compensate for the NumPy overhead.

**Qualitative take-away**: `jaxtra.scipy.linalg.ldl` is the right choice when
you need the factorization itself (e.g. to extract the D matrix or permutation
for downstream use), or when batching many small systems via `vmap`. For a
single solve `jax.scipy.linalg.solve` remains faster until `ldl_solve` gains
a JIT-compatible implementation.
