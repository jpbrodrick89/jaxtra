"""Direct LAPACK ORMQR wrappers via SciPy.

Provides ``ormqr_lapack``, which calls one of the four LAPACK routines:
  sormqr  (float32, real)
  dormqr  (float64, real)
  cunmqr  (complex64)
  zunmqr  (complex128)

The function handles workspace allocation, Fortran-order requirements, and
batch dimensions.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg.lapack as _sl

# ---------------------------------------------------------------------------
# LAPACK routine dispatch
# ---------------------------------------------------------------------------

_ORMQR_FNS: dict[np.dtype, object] = {
    np.dtype("float32"):    _sl.sormqr,
    np.dtype("float64"):    _sl.dormqr,
    np.dtype("complex64"):  _sl.cunmqr,
    np.dtype("complex128"): _sl.zunmqr,
}

_SUPPORTED_DTYPES = tuple(_ORMQR_FNS.keys())


def _get_fn(dtype: np.dtype):
    fn = _ORMQR_FNS.get(np.dtype(dtype))
    if fn is None:
        raise TypeError(
            f"ormqr_lapack does not support dtype {dtype!r}. "
            f"Supported dtypes: {list(_ORMQR_FNS.keys())}"
        )
    return fn


# ---------------------------------------------------------------------------
# Single-matrix kernel (no batch dimension)
# ---------------------------------------------------------------------------

def _ormqr_single(
    a: np.ndarray,   # (m, k) Householder reflectors
    tau: np.ndarray, # (k,)   Householder scalars
    c: np.ndarray,   # (m, n) or (n, m) depending on side
    side: bytes,
    trans: bytes,
) -> np.ndarray:
    """Call LAPACK ormqr/unmqr for a single (non-batched) problem."""
    fn = _get_fn(a.dtype)

    # LAPACK expects Fortran-contiguous ('F') arrays.
    a_f = np.asfortranarray(a)
    c_out = np.asfortranarray(c.copy())

    # Workspace query (lwork = -1)
    cq, work_q, info = fn(side, trans, a_f, tau, c_out, -1)
    if info != 0:
        raise RuntimeError(f"LAPACK ormqr workspace query failed (info={info})")
    lwork = max(1, int(np.real(work_q[0])))

    # Actual computation
    cq, _, info = fn(side, trans, a_f, tau, c_out, lwork, overwrite_c=1)
    if info != 0:
        raise RuntimeError(f"LAPACK ormqr failed (info={info})")

    return cq


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ormqr_lapack(
    a: np.ndarray,
    tau: np.ndarray,
    c: np.ndarray,
    *,
    left: bool = True,
    transpose: bool = False,
) -> np.ndarray:
    """Multiply a matrix by Q from a QR factorization using LAPACK.

    Computes one of the four operations:

    =========  =========  =============
    ``left``   ``trans``  Operation
    =========  =========  =============
    ``True``   ``False``  ``Q   @ C``
    ``True``   ``True``   ``Qᴴ  @ C``
    ``False``  ``False``  ``C   @ Q``
    ``False``  ``True``   ``C   @ Qᴴ``
    =========  =========  =============

    For real arrays ``Qᴴ = Qᵀ``; for complex arrays the conjugate
    transpose is used.

    Parameters
    ----------
    a:
        Householder reflectors as returned by :func:`numpy.linalg.qr` (raw
        mode) or SciPy's ``geqrf``/``geqp3``.  Shape ``[..., m, k]``.
    tau:
        Householder scalar factors, shape ``[..., k]``.
    c:
        Matrix to multiply.  Shape ``[..., m, n]`` when ``left=True`` or
        ``[..., n, m]`` when ``left=False``.
    left:
        If ``True`` multiply on the left (``Q @ C``); otherwise on the right
        (``C @ Q``).
    transpose:
        If ``True`` use ``Qᵀ`` / ``Qᴴ`` instead of ``Q``.

    Returns
    -------
    numpy.ndarray
        Result array with the same shape as *c*.

    Raises
    ------
    TypeError
        If *dtype* is not one of float32, float64, complex64, complex128.
    ValueError
        If shapes are incompatible.
    RuntimeError
        If the underlying LAPACK call returns a non-zero info code.
    """
    a = np.asarray(a)
    tau = np.asarray(tau)
    c = np.asarray(c)

    # Promote integer inputs to float64.
    if not np.issubdtype(a.dtype, np.floating) and not np.issubdtype(
        a.dtype, np.complexfloating
    ):
        a = a.astype(np.float64)
        tau = tau.astype(np.float64)
        c = c.astype(np.float64)

    # Ensure consistent dtype across inputs.
    dtype = np.result_type(a, tau, c)
    if dtype not in _ORMQR_FNS:
        # Upcast to nearest supported type.
        if np.issubdtype(dtype, np.complexfloating):
            dtype = np.complex128
        else:
            dtype = np.float64
    a = a.astype(dtype)
    tau = tau.astype(dtype)
    c = c.astype(dtype)

    # LAPACK side / trans characters.
    side: bytes = b"L" if left else b"R"
    is_complex = np.issubdtype(dtype, np.complexfloating)
    if transpose:
        trans: bytes = b"C" if is_complex else b"T"
    else:
        trans = b"N"

    # -----------------------------------------------------------------------
    # Shape validation
    # -----------------------------------------------------------------------
    if a.ndim < 2:
        raise ValueError(f"'a' must have at least 2 dimensions, got shape {a.shape}")
    if tau.ndim < 1:
        raise ValueError(f"'tau' must have at least 1 dimension, got shape {tau.shape}")
    if c.ndim < 2:
        raise ValueError(f"'c' must have at least 2 dimensions, got shape {c.shape}")

    m = a.shape[-2]  # rows of the Householder matrix
    k = tau.shape[-1]

    if left and c.shape[-2] != m:
        raise ValueError(
            f"ormqr with left=True requires c.shape[-2] == a.shape[-2], "
            f"but got a.shape={a.shape} and c.shape={c.shape}."
        )
    if not left and c.shape[-1] != m:
        raise ValueError(
            f"ormqr with left=False requires c.shape[-1] == a.shape[-2], "
            f"but got a.shape={a.shape} and c.shape={c.shape}."
        )

    # -----------------------------------------------------------------------
    # Batch dispatch
    # -----------------------------------------------------------------------
    batch_shape = np.broadcast_shapes(a.shape[:-2], tau.shape[:-1], c.shape[:-2])

    if len(batch_shape) == 0:
        # Non-batched: a is (m, k), tau is (k,), c is (c_rows, c_cols).
        return _ormqr_single(a, tau, c, side, trans)

    # Broadcast to common batch shape and iterate.
    a_b = np.broadcast_to(a, batch_shape + a.shape[-2:])
    tau_b = np.broadcast_to(tau, batch_shape + tau.shape[-1:])
    c_b = np.broadcast_to(c, batch_shape + c.shape[-2:])

    out = np.empty(batch_shape + c.shape[-2:], dtype=dtype)
    for idx in np.ndindex(batch_shape):
        out[idx] = _ormqr_single(
            a_b[idx], tau_b[idx], c_b[idx], side, trans
        )
    return out
