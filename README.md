# jaxtra

Native JAX extensions for LAPACK routines and GPU-accelerated linear algebra — no jaxlib rebuild required.

Built on JAX's XLA Foreign Function Interface (FFI): C++/LAPACK (and, soon, CUDA) kernels registered at runtime as proper JAX primitives — JIT, vmap, and grad compatible.

---

## Why jaxtra?

JAX exposes only a small slice of LAPACK and provides no public API for custom GPU lowerings.
`jaxtra` fills the gap by shipping pre-built XLA FFI kernels that plug directly into the JAX primitive system, matching the calling conventions of `jax.lax.linalg` and `jax.scipy.linalg`.

GPU lowerings are under active development and will land shortly.

---

## Quick start

```bash
pip install jaxtra          # or: uv add jaxtra
```

```python
import jax.numpy as jnp
from jax._src.lax.linalg import geqrf
from jaxtra import ormqr

A = jnp.array([[1, 2], [3, 4], [5, 6]], dtype=jnp.float64)
b = jnp.ones((3, 1), dtype=jnp.float64)
H, taus = geqrf(A)                                    # compact QR
Qtb = ormqr(H, taus, b, left=True, transpose=True)   # Qᵀ @ b, no Q formed
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
