"""
jaxtra.lax.linalg — JAX lax.linalg-style interface.

Exposes:

* ``geqrf``           — QR factorisation returning compact Householder vectors + taus
* ``geqp3``           — column-pivoted QR factorisation (JAX built-in)
* ``ormqr``           — apply Q (without forming it) to a matrix via a native XLA FFI kernel
* ``householder_product`` — form Q explicitly from Householder vectors (JAX built-in)

The naming and calling convention mirrors ``jax.lax.linalg``.
"""
from __future__ import annotations

# Re-export JAX's existing primitives.
from jax._src.lax.linalg import geqrf, geqp3, householder_product  # noqa: F401

# Our new XLA FFI primitive.
from jaxtra._core import ormqr, ormqr_p  # noqa: F401

__all__ = ["geqrf", "geqp3", "householder_product", "ormqr", "ormqr_p"]
