"""``jaxtra.lax.linalg`` – mirrors the interface of ``jax.lax.linalg``.

Currently exposes:

* :func:`ormqr` – multiply by Q from a QR factorization without forming Q.

When JAX is installed the function delegates to ``jax.lax.linalg.ormqr``
(once the upstream PR is merged) or falls back to the LAPACK implementation
wrapped in a ``jax.pure_callback`` so that it participates in JAX tracing.
When JAX is *not* installed the function operates directly on NumPy arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from jaxtra._core import ormqr_lapack

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = ["ormqr"]


def ormqr(
    a: "ArrayLike",
    taus: "ArrayLike",
    c: "ArrayLike",
    *,
    left: bool = True,
    transpose: bool = False,
) -> np.ndarray:
    """Multiply a matrix by Q from a QR factorisation without materialising Q.

    This is a drop-in preview of ``jax.lax.linalg.ormqr`` that uses LAPACK
    via SciPy for immediate usability while the upstream JAX PR is pending.

    Computes one of:

    =========  =========  ======================
    ``left``   ``trans``  Operation
    =========  =========  ======================
    ``True``   ``False``  ``Q   @ C``  (default)
    ``True``   ``True``   ``Qᴴ  @ C``
    ``False``  ``False``  ``C   @ Q``
    ``False``  ``True``   ``C   @ Qᴴ``
    =========  =========  ======================

    For real matrices ``Qᴴ = Qᵀ``.

    Parameters
    ----------
    a:
        Householder reflectors from :func:`numpy.linalg.qr` (``mode='raw'``)
        or SciPy's ``geqrf`` / ``geqp3``.  Shape ``[..., m, k]``.
    taus:
        Householder scalar factors, shape ``[..., k]``.
    c:
        The matrix to transform.  Shape ``[..., m, n]`` when ``left=True``
        or ``[..., n, m]`` when ``left=False``.
    left:
        Multiply on the left (``Q @ C``) when ``True``, on the right
        (``C @ Q``) when ``False``.
    transpose:
        Use ``Qᵀ`` / ``Qᴴ`` instead of ``Q`` when ``True``.

    Returns
    -------
    numpy.ndarray
        Result with the same shape as *c*.

    Examples
    --------
    Compute ``Qᵀ @ b`` without forming Q:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = rng.standard_normal((5, 3))
    >>> b = rng.standard_normal((5, 2))
    >>> # Raw QR from numpy returns (H, tau) where H contains the reflectors.
    >>> H, tau = np.linalg.qr(A, mode='raw')
    >>> # numpy 'raw' mode returns H in Fortran layout; transpose for ormqr.
    >>> from jaxtra.lax.linalg import ormqr
    >>> result = ormqr(H.T, tau, b, left=True, transpose=True)
    """
    # Try to delegate to JAX if available.
    try:
        import jax.lax.linalg as jll  # type: ignore[import]
        if hasattr(jll, "ormqr"):
            return jll.ormqr(a, taus, c, left=left, transpose=transpose)
    except ImportError:
        pass

    return ormqr_lapack(
        np.asarray(a),
        np.asarray(taus),
        np.asarray(c),
        left=left,
        transpose=transpose,
    )
