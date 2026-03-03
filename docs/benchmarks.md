# Benchmarks

## Pentadiagonal solve

Comparison of `pentadiagonal_solve` (LAPACK `gbsv`) against dense LU and
scipy's banded Cholesky for SPD pentadiagonal systems (float64, CPU):

| Method                 | Implementation                                                  |
| ---------------------- | --------------------------------------------------------------- |
| **jaxtra (gbsv)**      | `pentadiagonal_solve` — LAPACK `dgbsv`, O(kn) banded LU (k=2) |
| **dense LU**           | `jnp.linalg.solve` — O(n³) dense LU                           |
| **scipy banded**       | `scipy.linalg.solveh_banded` — O(kn) banded Cholesky          |

All JAX timings use `jax.jit` + `jax.block_until_ready` with two warmup runs
followed by five timed repetitions; the reported value is the median.
Raw results are in `benchmarks/results/bench_banded.csv`.

```{figure} ../benchmarks/results/bench_banded.png
:alt: Benchmark: pentadiagonal solve
:width: 90%
:align: center
```

---

## ORMQR least-squares

Comparison of jaxtra's ORMQR-based least-squares solver against two
alternatives on an overdetermined system **A x ≈ b** (float64, CPU):

| Method             | Implementation                                              |
| ------------------ | ----------------------------------------------------------- |
| **jaxtra (ORMQR)** | `qr_multiply` — LAPACK `dormqr`, Q never formed             |
| **dense QR**       | `jnp.linalg.qr` — Q explicitly materialised, then `Q.T @ b` |
| **scipy SVD**      | `scipy.linalg.lstsq` — SVD-based, NumPy arrays              |

All JAX timings use `jax.jit` + `jax.block_until_ready` with one warmup run
followed by five timed repetitions; the reported value is the median.
Raw results are in `benchmarks/results/bench_least_squares.csv`.

## Results

### 20 columns

```{figure} ../benchmarks/results/bench_cols20.png
:alt: Benchmark: 20 columns
:width: 90%
:align: center
```

### 50 columns

```{figure} ../benchmarks/results/bench_cols50.png
:alt: Benchmark: 50 columns
:width: 90%
:align: center
```

### 100 columns

```{figure} ../benchmarks/results/bench_cols100.png
:alt: Benchmark: 100 columns
:width: 90%
:align: center
```

## Summary

Speedups are largest for tall matrices with many columns — the regime where
forming Q explicitly wastes the most memory and arithmetic.

**vs. dense QR (jnp.linalg.qr)**
: jaxtra avoids materialising the full M × K Q matrix. At 100 columns and
5 000 – 20 000 rows, the typical speedup is **1.7 – 2.5×**. At smaller
column counts the gap narrows (Q is cheaper to form) but jaxtra remains
faster or equal across all tested sizes.

**vs. scipy lstsq (SVD)**
: SVD computes all singular values and vectors, which is O(MN²) + O(N³) vs.
ORMQR's O(MN²) with a smaller constant and no Q allocation. At 100 columns
the speedup ranges from **2 – 3.4×** and grows with matrix size, reaching
**2.8×** at 20 000 × 100.

**Qualitative take-away**: for the common case of tall, skinny systems
(M ≫ N, N ≈ 50 – 200) jaxtra's `qr_multiply` is the fastest JAX-native
option and is fully `jit` / `vmap` / `grad` compatible.
