# jaxtra

**Native JAX extensions for LAPACK routines and GPU-accelerated linear algebra via XLA FFI.**

jaxtra ships XLA FFI kernels registered at runtime as proper JAX primitives — fully compatible with `jit`, `vmap`, and `grad` — without requiring a jaxlib rebuild.

:::{note}
jaxtra is a forward-compatibility shim for [JAX PR #35104](https://github.com/google/jax/pull/35104).
Once that PR merges, this package can be replaced by a straight import from `jax.lax.linalg`.
:::

## Quick start

```python
import jax.numpy as jnp
from jax._src.lax.linalg import geqrf
from jaxtra import ormqr

A = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
b = jnp.ones((3, 1))

H, taus = geqrf(A)           # compact QR
Qtb = ormqr(H, taus, b, left=True, transpose=True)  # Qᵀ @ b, no Q formed
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
```
