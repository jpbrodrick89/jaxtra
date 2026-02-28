"""``jaxtra.scipy.linalg`` – mirrors the interface of ``jax.scipy.linalg``.

Currently exposes:

* :func:`qr_multiply` – combined QR decomposition + Q-multiply step.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from jaxtra._numpy_lapack import ormqr_lapack

import scipy.linalg.lapack as _sl

__all__ = ["qr_multiply"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _geqrf(a: np.ndarray):
    """Call LAPACK geqrf and return (QR-factor array, tau)."""
    dtype = a.dtype
    if np.issubdtype(dtype, np.complexfloating):
        if dtype == np.complex64:
            fn = _sl.cgeqrf
        else:
            fn = _sl.zgeqrf
    else:
        if dtype == np.float32:
            fn = _sl.sgeqrf
        else:
            fn = _sl.dgeqrf

    r, tau, _, info = fn(a)
    if info != 0:
        raise RuntimeError(f"geqrf failed (info={info})")
    return r, tau


def _geqp3(a: np.ndarray):
    """Call LAPACK geqp3 (pivoted QR) and return (r, tau, jpvt)."""
    dtype = a.dtype
    if np.issubdtype(dtype, np.complexfloating):
        if dtype == np.complex64:
            fn = _sl.cgeqp3
        else:
            fn = _sl.zgeqp3
    else:
        if dtype == np.float32:
            fn = _sl.sgeqp3
        else:
            fn = _sl.dgeqp3

    r, jpvt_out, tau, _, info = fn(a)
    if info != 0:
        raise RuntimeError(f"geqp3 failed (info={info})")
    return r, tau, jpvt_out


def _promote_inexact(*arrays: np.ndarray):
    """Promote all arrays to a common floating/complex dtype."""
    dtype = np.result_type(*[a.dtype for a in arrays])
    if not (np.issubdtype(dtype, np.floating) or
            np.issubdtype(dtype, np.complexfloating)):
        dtype = np.float64
    # Promote to at least float32.
    if dtype == np.float16:
        dtype = np.float32
    return tuple(a.astype(dtype) for a in arrays)


# ---------------------------------------------------------------------------
# Public overloads (for type checkers)
# ---------------------------------------------------------------------------

@overload
def qr_multiply(
    a: np.ndarray, c: np.ndarray, mode: str = "right",
    pivoting: Literal[False] = False, conjugate: bool = False,
    overwrite_a: bool = False, overwrite_c: bool = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def qr_multiply(
    a: np.ndarray, c: np.ndarray, mode: str = "right",
    pivoting: Literal[True] = True, conjugate: bool = False,
    overwrite_a: bool = False, overwrite_c: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def qr_multiply(
    a: np.ndarray,
    c: np.ndarray,
    mode: str = "right",
    pivoting: bool = False,
    conjugate: bool = False,
    overwrite_a: bool = False,
    overwrite_c: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QR decomposition combined with a Q-multiply step.

    A preview of :func:`jax.scipy.linalg.qr_multiply` implemented via LAPACK
    through SciPy so it can be used before the upstream PR is merged.

    Computes the QR decomposition of *a* and multiplies *c* by the resulting
    Q matrix without ever materialising Q.

    Parameters
    ----------
    a:
        Matrix to decompose, shape ``(..., M, N)``.
    c:
        Matrix (or 1-D vector) to multiply by Q.

        - ``mode='left'``:  shape ``(..., K, P)`` where ``K = min(M, N)``.
          A 1-D array is treated as a column vector of length ``K``.
        - ``mode='right'``: shape ``(..., P, M)``.
          A 1-D array is treated as a row vector of length ``M``.

        In both cases a 1-D result is returned for 1-D input.
    mode:
        ``'right'`` (default) computes ``c @ Q`` and returns the result
        truncated to ``K = min(M, N)`` columns.  ``'left'`` computes ``Q @ c``.
    pivoting:
        If ``True``, perform column-pivoted QR (geqp3) and return a third
        element containing 0-based permutation indices.
    conjugate:
        If ``True``, apply the element-wise complex conjugate of Q (i.e. use
        ``conj(Q)`` rather than Q).  For real arrays this has no effect.
    overwrite_a, overwrite_c:
        Ignored (present for API compatibility with
        :func:`scipy.linalg.qr_multiply`).

    Returns
    -------
    result : numpy.ndarray
        ``Q @ c`` or ``c @ Q`` (possibly conjugated), same leading batch
        dimensions as the inputs.
    R : numpy.ndarray
        Upper-triangular factor, shape ``(..., K, N)``.
    P : numpy.ndarray
        *(only when* ``pivoting=True`` *)* 0-based column permutation indices,
        shape ``(..., N,)``.

    Examples
    --------
    Least-squares via ``Qᵀb`` and back-substitution:

    >>> import numpy as np
    >>> from jaxtra.scipy.linalg import qr_multiply
    >>> A = np.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
    >>> b = np.array([2., 4., 5., 4.])
    >>> Qtb, R = qr_multiply(A, b, mode='right')
    >>> x = np.linalg.solve(R, Qtb)   # back-substitution for square R
    """
    del overwrite_a, overwrite_c  # unused

    a, c = np.asarray(a), np.asarray(c)
    a, c = _promote_inexact(a, c)

    if mode not in ("right", "left"):
        raise ValueError(f"mode must be 'right' or 'left', got {mode!r}")

    onedim = c.ndim == 1
    if onedim:
        c = c[:, np.newaxis] if mode == "left" else c[np.newaxis, :]

    m, n = a.shape[-2], a.shape[-1]
    k = min(m, n)

    if mode == "left":
        if c.shape[-2] != k:
            raise ValueError(
                f"Incompatible shapes for Q @ c: a has shape {a.shape} so "
                f"Q has {k} columns, but c has {c.shape[-2]} rows (expected {k})."
            )
    else:
        if c.shape[-1] != m:
            raise ValueError(
                f"Incompatible shapes for c @ Q: a has shape {a.shape} so "
                f"Q has {m} rows, but c has {c.shape[-1]} columns (expected {m})."
            )

    # -----------------------------------------------------------------------
    # QR factorization (possibly pivoted)
    # -----------------------------------------------------------------------
    batch_shape = a.shape[:-2]
    if len(batch_shape) == 0:
        # Non-batched path.
        if pivoting:
            r, tau, jpvt_out = _geqp3(a)
            p = jpvt_out - 1  # 1-based → 0-based
        else:
            r, tau = _geqrf(a)
            p = None
    else:
        # Batched path: loop over batch dimensions.
        a_flat = a.reshape(-1, m, n)
        r_list, tau_list = [], []
        p_list = []
        for i in range(a_flat.shape[0]):
            if pivoting:
                ri, ti, pi = _geqp3(a_flat[i])
                r_list.append(ri)
                tau_list.append(ti)
                p_list.append(pi - 1)
            else:
                ri, ti = _geqrf(a_flat[i])
                r_list.append(ri)
                tau_list.append(ti)
        r = np.stack(r_list).reshape(batch_shape + (m, n))
        tau = np.stack(tau_list).reshape(batch_shape + (min(m, n),))
        p = np.stack(p_list).reshape(batch_shape + (n,)) if pivoting else None

    # -----------------------------------------------------------------------
    # Pad c if needed (left mode, m > n: Q is m×m but c may be k×P).
    # -----------------------------------------------------------------------
    if m > n and mode == "left":
        pad_shape = c.shape[:-2] + (m - k,) + c.shape[-1:]
        zeros = np.zeros(pad_shape, dtype=c.dtype)
        c = np.concatenate([c, zeros], axis=-2)

    # -----------------------------------------------------------------------
    # conjugate swaps left/right because transposing C swaps which side Q is on.
    # -----------------------------------------------------------------------
    if conjugate:
        c = c.swapaxes(-1, -2)

    left = (mode == "left") != conjugate
    cQ = ormqr_lapack(r, tau, c, left=left, transpose=conjugate)

    if conjugate:
        cQ = cQ.swapaxes(-1, -2)

    if mode == "right":
        cQ = cQ[..., :k]

    if onedim:
        cQ = cQ.ravel()

    # Upper-triangular R, trimmed to k rows.
    r_upper = np.triu(r[..., :k, :])

    if pivoting:
        return cQ, r_upper, p
    return cQ, r_upper
