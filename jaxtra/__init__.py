"""jaxtra: LAPACK ORMQR exposed with a JAX-compatible API.

This package provides ``ormqr`` (orthogonal QR multiply) and
``qr_multiply`` before the upstream JAX PR is merged, using LAPACK
routines (sormqr / dormqr / cunmqr / zunmqr) via SciPy.
"""

from jaxtra._core import ormqr_lapack  # noqa: F401
