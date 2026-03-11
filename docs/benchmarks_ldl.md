# LDL Benchmarks

Comparison of jaxtra's LDL (Bunch-Kaufman) factorisation against LU for a
full solve (factorization + triangular solve) of a single 1-D right-hand side,
on square matrices (float64 and complex128, CPU, n = 50 -- 5000):

| Method            | Implementation                                                               |
| ----------------- | ---------------------------------------------------------------------------- |
| **LDL full solve**| `jaxtra._src.lax.linalg.ldl` (JIT) + `ldl_solve` via LAPACK `sytrs` (JIT)  |
| **LU full solve** | `jax.scipy.linalg.solve` -- fully JIT-compiled LAPACK `dgetrf`/`dtrsv`       |

All timings use `jax.block_until_ready` with one warmup run followed by five
timed repetitions; the reported value is the median.
Raw results are in `benchmarks/results/bench_ldl.csv`.

**Note on D storage**: D is returned as a full `(n, n)` block-diagonal matrix,
not a 1-D array. Bunch-Kaufman pivoting can produce 2x2 off-diagonal blocks in
D, so a 1-D diagonal representation is insufficient. This matches scipy's
behavior.

## Results

```{figure} ../benchmarks/results/bench_ldl.png
:alt: LDL vs LU full solve benchmark
:width: 95%
:align: center
```

## Summary

**Performance is LAPACK-implementation dependent.** The relative speed of
`sytrf` (LDL) vs `getrf` (LU) varies significantly across BLAS/LAPACK vendors:

- **OpenBLAS / MKL** (Linux, GHA): `sytrf` exploits symmetry and is
  **1.2 -- 1.7x faster** than `getrf` for both real and complex types at
  n >= 500.
- **Apple Accelerate** (macOS): `getrf` is heavily optimized (AMX/NEON) while
  `sytrf` is not, so LU can be faster for real symmetric solves. Complex
  Hermitian LDL still outperforms LU (1.2 -- 2.2x).

**When to use LDL over LU:**

- Complex Hermitian indefinite systems (consistently faster across platforms).
- When you need the factorization itself (e.g. to extract D, the permutation,
  or to solve multiple right-hand sides).
- When batching via `vmap`.
- On Linux with OpenBLAS/MKL, LDL is faster for real symmetric systems too.
