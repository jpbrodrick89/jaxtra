# jaxtra

**JAX extensions for LAPACK routines and GPU-accelerated linear algebra.**

Registers XLA FFI kernels as proper JAX primitives — fully compatible with `jit`, `vmap`, and `grad` — backed by LAPACK (CPU) and cuSOLVER (GPU).

## Quick start

```python
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import jaxtra.scipy.linalg as jsla

# Solve a least-squares problem A @ x ≈ b via QR.
A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
b = jnp.array([2., 4., 5., 4.])

# qr_multiply decomposes A and applies Qᵀ to b in one step — no Q formed.
Qtb, R = jsla.qr_multiply(A, b, mode='right')
x = jsl.solve_triangular(R, Qtb)
```

## Installation

```bash
pip install jaxtra            # CPU (LAPACK via SciPy at runtime)
pip install "jaxtra[gpu]"     # + NVIDIA cuSOLVER runtime libs
```

## Modules

```{toctree}
:maxdepth: 2

api
benchmarks
```
