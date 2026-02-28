"""jaxtra: LAPACK ORMQR as a native JAX extension (XLA FFI).

Provides ``ormqr`` — apply Q from a compact QR factorisation to a matrix C
**without ever forming Q** — as a proper JAX primitive backed by a C++/LAPACK
kernel registered through JAX's XLA Foreign Function Interface.

The key difference from ``jnp.linalg.qr``:
  • ``jnp.linalg.qr`` materialises the full Q matrix (expensive for tall matrices).
  • ``jaxtra.ormqr`` takes the Householder vectors and taus directly and applies
    Q implicitly via LAPACK dormqr / sormqr / cunmqr / zunmqr.

Quick start
-----------
>>> import jax.numpy as jnp
>>> from jax._src.lax.linalg import geqrf
>>> from jaxtra import ormqr
>>>
>>> A = jnp.array([[1,2],[3,4],[5,6]], dtype=jnp.float64)
>>> b = jnp.ones((3, 1), dtype=jnp.float64)
>>> H, taus = geqrf(A)            # compact QR — taus are first-class JAX arrays
>>> Qtb = ormqr(H, taus, b, left=True, transpose=True)   # Qᵀ @ b, no Q formed
"""

# XLA FFI-backed JAX primitive (JIT-compatible, vmap-compatible).
from jaxtra._core import ormqr  # noqa: F401

# Pure numpy/scipy fallback (no JAX required, useful for non-JIT contexts).
from jaxtra._numpy_lapack import ormqr_lapack  # noqa: F401
