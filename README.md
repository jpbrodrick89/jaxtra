# jaxtra

Native JAX extensions for LAPACK routines and GPU-accelerated linear algebra — no jaxlib rebuild required.

Built on JAX's XLA Foreign Function Interface (FFI): C++/LAPACK and cuSolver kernels registered at runtime as proper JAX primitives — JIT, vmap, and grad compatible.

---

## Why jaxtra?

JAX does not expose every LAPACK routine or every NVIDIA library function.
`jaxtra` exposes the missing ones that are practically useful, shipping them as proper JAX primitives that plug directly into the JAX primitive system and match the calling conventions of `jax.lax.linalg` and `jax.scipy.linalg`.

---

## Quick start

```bash
pip install jaxtra          # or: uv add jaxtra
```

GPU kernels (cuSolver) are built separately — see [GPU support](#gpu-support) below.

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

---

## Reference

### LAPACK routines exposed

| Routine                    | Description                                                                                     | Primitive                                   |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `?ormqr` (`s`/`d`/`c`/`z`) | Multiply a matrix by Q (or Qᵀ/Qᴴ) from a compact Householder QR factorisation without forming Q | `jaxtra._src.lax.linalg.ormqr`              |
| `?gbsv`  (`s`/`d`/`c`/`z`) | Solve a pentadiagonal (banded, KL=KU=2) linear system via banded LU; GPU uses cuSPARSE `gpsvInterleavedBatch` | `jaxtra.lax.linalg.pentadiagonal_solve` |

---

## GPU support

Install with the `[gpu]` extra to pull in the NVIDIA runtime libraries and build the cuSolver extension:

```bash
pip install jaxtra[gpu]          # pulls nvidia-cusolver-cu12 + builds _jaxtra_cuda.so if CUDA toolkit is present
# or with uv:
uv add jaxtra[gpu]
```

The build auto-detects the CUDA toolkit: `_jaxtra_cuda.so` is compiled when `nvcc`/`CUDAToolkit` is found, skipped silently otherwise. If you need to override:

```bash
JAXTRA_CUDA=OFF  pip install jaxtra[gpu]   # force CPU-only even if CUDA is present
JAXTRA_CUDA=ON   pip install jaxtra[gpu]   # require CUDA; fail if not found
```

Once `_jaxtra_cuda.so` is present alongside `_jaxtra.so`, `jaxtra` detects and loads it automatically at import time — no code change required.

Requirements: CUDA ≥ 11.x, cuSolver, Abseil (fetched automatically via CMake `FetchContent` if not installed system-wide).

---

### Public API

#### `jaxtra.scipy.linalg`

Mirrors `jax.scipy.linalg`.

| Symbol        | Description                                                                              |
| ------------- | ---------------------------------------------------------------------------------------- |
| `qr_multiply` | Combined QR decomposition and Q-multiply in one step; mirrors `scipy.linalg.qr_multiply` |

The underlying `ormqr` primitive is accessible at `jaxtra._src.lax.linalg.ormqr`
for users who need to apply Q directly (same convention as reaching into
`jax._src.lax.linalg` for primitives not yet in the public API).

#### `jaxtra.lax.linalg`

Low-level primitives analogous to `jax.lax.linalg`.

| Symbol                | Description                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `pentadiagonal_solve` | Solve A x = b for a pentadiagonal matrix A (five diagonals). CPU: LAPACK `gbsv`. GPU: cuSPARSE `gpsvInterleavedBatch`. Supports `jit`, `vmap`, `grad`. |
