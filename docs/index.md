# jaxtra

**JAX extensions for LAPACK routines and GPU-accelerated linear algebra.**

Registers XLA FFI kernels as proper JAX primitives — fully compatible with `jit`, `vmap`, and `grad` — backed by LAPACK (CPU) and cuSOLVER/cuSPARSE (GPU).

## Functions

```{toctree}
:hidden:

api
```

| Function | Description |
|----------|-------------|
| {func}`jaxtra.scipy.linalg.qr_multiply` | QR decomposition and Q-multiply in one step |
| {func}`jaxtra.lax.linalg.pentadiagonal_solve` | Pentadiagonal linear solve |
| {func}`jaxtra.lax.linalg.pentadiagonal_solveh` | Hermitian pentadiagonal linear solve |
| {func}`jaxtra._src.lax.linalg.ormqr` | Multiply by Q from a QR factorization |

## Installation

```bash
pip install jaxtra             # CPU (LAPACK via SciPy at runtime)
pip install jaxtra[cuda12]     # + CUDA 12 support
pip install jaxtra[cuda13]     # + CUDA 13 support
```

## Quick example

```python
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jaxtra.scipy.linalg as jslx

# Solve a least-squares problem A @ x ≈ b via QR.
key_a, key_b = jax.random.split(jax.random.key(0))
A = jax.random.normal(key_a, (5000, 100))   # tall, skinny matrix
b = jax.random.normal(key_b, (5000,))

# qr_multiply decomposes A and applies Qᵀ to b in one step — no Q formed.
Qtb, R = jslx.qr_multiply(A, b, mode='right')
x = jsl.solve_triangular(R, Qtb)
```
