# LDL Benchmarks

Comparison of jaxtra's LDL (Bunch-Kaufman) factorisation against LU and Cholesky
on square matrices (float64 and complex128, CPU):

| Method        | Implementation                                                   |
| ------------- | ---------------------------------------------------------------- |
| **LDL**       | `jaxtra.scipy.linalg.ldl` — LAPACK `dsytrf` / `zhetrf`          |
| **LU**        | `jax._src.lax.linalg.lu` — LAPACK `dgetrf`, always applicable   |
| **Cholesky**  | `jax._src.lax.linalg.cholesky` — PSD matrices only (f64 series) |

All timings use `jax.jit` + `jax.block_until_ready` with one warmup run
followed by five timed repetitions; the reported value is the median.
Raw results are in `benchmarks/results/bench_ldl.csv`.

## Results

```{figure} ../benchmarks/results/bench_ldl.png
:alt: LDL vs LU vs Cholesky benchmark
:width: 95%
:align: center
```

## Summary

**LDL vs LU (real symmetric f64)**
: LDL is faster than LU at small sizes (n ≤ 200) and for large matrices
(n ≥ 2 000) where the Bunch-Kaufman pivot strategy requires fewer operations
than full partial pivoting. At intermediate sizes (n ≈ 500 – 1 000) LU is
faster due to LAPACK's highly optimised panel factorisation outperforming
sytrf's blocked algorithm.

**LDL vs LU (complex Hermitian c128)**
: LDL (hetrf) is consistently **1.2 – 1.6×** faster than LU across all sizes,
with the advantage growing for large matrices. Complex Hermitian structure
allows sytrf to halve the memory traffic vs. getrf.

**Cholesky (PSD baseline)**
: Cholesky is the fastest option when the matrix is known to be positive
definite — it avoids pivoting entirely. LDL is the correct choice for
indefinite symmetric/Hermitian matrices where Cholesky would fail.

**Qualitative take-away**: use `jaxtra.scipy.linalg.ldl` for symmetric or
Hermitian indefinite systems. It is fully `jit` / `vmap` / `grad` compatible
and consistently outperforms generic LU factorisation for large Hermitian
problems.
