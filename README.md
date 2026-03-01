# jaxtra

Native JAX extensions for LAPACK routines and GPU-accelerated linear algebra — no jaxlib rebuild required.

Built on JAX's XLA Foreign Function Interface (FFI): C++/LAPACK (and, soon, CUDA) kernels registered at runtime as proper JAX primitives — JIT, vmap, and grad compatible.

---

## Why jaxtra?

JAX does not expose every LAPACK routine or every NVIDIA library function.
`jaxtra` exposes the missing ones that are practically useful, shipping them as proper JAX primitives that plug directly into the JAX primitive system and match the calling conventions of `jax.lax.linalg` and `jax.scipy.linalg`.

GPU lowerings are under active development and will land shortly.

---

## Quick start

```bash
pip install jaxtra          # or: uv add jaxtra
```

```python
import jax.numpy as jnp
import jaxtra.scipy.linalg as jsla

# Solve a least-squares problem A @ x ≈ b via QR.
A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
b = jnp.array([2., 4., 5., 4.])

# qr_multiply decomposes A and applies Qᵀ to b in one step — no Q formed.
Qtb, R = jsla.qr_multiply(A, b, mode='right')
x = jnp.linalg.solve(R, Qtb)
```

---

## Reference

### LAPACK routines exposed

| Routine | Description | Primitive |
|---------|-------------|-----------|
| `?ormqr` (`s`/`d`/`c`/`z`) | Multiply a matrix by Q (or Qᵀ/Qᴴ) from a compact Householder QR factorisation without forming Q | `jaxtra._core.ormqr` |

---

### Public API

#### `jaxtra.lax.linalg`

Mirrors the `jax.lax.linalg` naming convention.

| Symbol | Description |
|--------|-------------|
| `ormqr` | Apply Q from a compact QR factorisation to a matrix — XLA FFI primitive, JIT/vmap/grad compatible |

#### `jaxtra.scipy.linalg`

Mirrors `jax.scipy.linalg`.

| Symbol | Description |
|--------|-------------|
| `qr_multiply` | Combined QR decomposition and Q-multiply in one step; mirrors `scipy.linalg.qr_multiply` |
