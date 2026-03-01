"""jaxtra: JAX extensions for LAPACK routines and GPU-accelerated linear algebra.

Registers XLA FFI kernels as proper JAX primitives — JIT, vmap, and grad
compatible — backed by LAPACK (CPU) and cuSOLVER (GPU).

Currently exposed:
  • ``ormqr``       — apply Q (or Qᵀ/Qᴴ) from a compact Householder QR to a
                      matrix **without forming Q**, backed by LAPACK
                      dormqr/sormqr/cunmqr/zunmqr.
  • ``ormqr_lapack``— NumPy/SciPy fallback for the same operation (no JAX needed).

Quick start
-----------
>>> import jax.numpy as jnp
>>> from jax._src.lax.linalg import geqrf
>>> from jaxtra import ormqr
>>>
>>> A = jnp.array([[1,2],[3,4],[5,6]], dtype=jnp.float64)
>>> b = jnp.ones((3, 1), dtype=jnp.float64)
>>> H, taus = geqrf(A)            # compact QR
>>> Qtb = ormqr(H, taus, b, left=True, transpose=True)   # Qᵀ @ b, no Q formed
"""

# XLA FFI-backed JAX primitive (JIT-compatible, vmap-compatible).
from jaxtra._core import ormqr  # noqa: F401

# Pure numpy/scipy fallback (no JAX required, useful for non-JIT contexts).
from jaxtra._numpy_lapack import ormqr_lapack  # noqa: F401
