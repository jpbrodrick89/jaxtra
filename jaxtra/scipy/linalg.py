"""``jaxtra.scipy.linalg`` – mirrors the interface of ``jax.scipy.linalg``.

Currently exposes:

* :func:`qr_multiply` – combined QR decomposition + Q-multiply step.
"""

from __future__ import annotations

from typing import Literal, overload

import jax.numpy as jnp

from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike
from jax._src.lax.linalg import geqrf, geqp3

from jaxtra._src.lax.linalg import ormqr

__all__ = ["qr_multiply"]


@overload
def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: Literal[False] = False, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array]: ...

@overload
def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: Literal[True] = True, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array, Array]: ...

@overload
def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: bool = False, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...


def qr_multiply(a: ArrayLike, c: ArrayLike, mode: str = 'right',
                pivoting: bool = False, conjugate: bool = False,
                overwrite_a: bool = False, overwrite_c: bool = False
                ) -> tuple[Array, Array] | tuple[Array, Array, Array]:
  """Calculate the QR decomposition and multiply Q with a matrix.

  JAX implementation of :func:`scipy.linalg.qr_multiply`.

  Args:
    a: array of shape ``(..., M, N)``. Matrix to be decomposed.
    c: array to be multiplied by Q. For ``mode='left'``, ``c`` has shape
      ``(..., K, P)`` where ``K = min(M, N)``. For ``mode='right'``, ``c``
      has shape ``(..., P, M)``. 1-D arrays are supported: for
      ``mode='left'``, treated as a length-``K`` column vector; for
      ``mode='right'``, treated as a length-``M`` row vector. The result
      is raveled back to 1-D in either case.
    mode: ``'right'`` (default) or ``'left'``.

      - ``'left'``: compute ``Q @ c`` (or ``conj(Q) @ c`` if ``conjugate=True``)
        and return ``(Q @ c, R)`` with result shape ``(..., M, P)``
      - ``'right'``: compute ``c @ Q`` (or ``c @ conj(Q)`` if ``conjugate=True``)
        and return ``(c @ Q, R)`` with result shape ``(..., P, K)`` where
        ``K = min(M, N)``
    pivoting: Allows the QR decomposition to be rank-revealing. If ``True``, compute
      the column-pivoted QR decomposition and return permutation indices
      as a third element.
    conjugate: If ``True``, use ``conj(Q)`` (element-wise complex conjugate)
      instead of ``Q``. For real arrays this has no effect.
    overwrite_a: unused in JAX
    overwrite_c: unused in JAX

  Returns:
    If ``pivoting`` is ``False``: ``(result, R)``

    If ``pivoting`` is ``True``: ``(result, R, P)``

  See also:
    - :func:`jax.scipy.linalg.qr`: SciPy-style QR decomposition API
    - :func:`jax.lax.linalg.ormqr`: XLA-style Q-multiply primitive

  Examples:
    Use :func:`qr_multiply` to efficiently solve a least-squares problem.
    For an overdetermined system ``A @ x ≈ b``, pass ``b`` as a 1-D row
    via ``mode='right'`` to obtain ``Q^T @ b`` and ``R`` in one step:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
    >>> b = jnp.array([2., 4., 5., 4.])
    >>> Qtb, R = jax.scipy.linalg.qr_multiply(A, b, mode='right')
    >>> x = jax.scipy.linalg.solve_triangular(R, Qtb)
    >>> jnp.allclose(A.T @ A @ x, A.T @ b)
    Array(True, dtype=bool)
  """
  del overwrite_a, overwrite_c  # unused
  a, c = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(c))
  if mode not in ('right', 'left'):
    raise ValueError(f"mode must be 'right' or 'left', got {mode!r}")

  onedim = c.ndim == 1
  if onedim:
    c = c[:, None] if mode == 'left' else c[None, :]

  m, n = a.shape[-2:]
  k = min(m, n)

  if mode == 'left':
    if c.shape[-2] != k:
      raise ValueError(
          f"Array shapes are not compatible for Q @ c operation: "
          f"a has shape {tuple(a.shape)} so Q has {k} columns, "
          f"but c has {c.shape[-2]} rows (expected {k}).")
  else:
    if c.shape[-1] != m:
      raise ValueError(
          f"Array shapes are not compatible for c @ Q operation: "
          f"a has shape {tuple(a.shape)} so Q has {m} rows, "
          f"but c has {c.shape[-1]} columns (expected {m}).")

  if pivoting:
    jpvt = jnp.zeros(a.shape[:-2] + (n,), dtype=jnp.int32)
    r, p, taus = geqp3(a, jpvt)
    p -= 1  # Convert geqp3's 1-based indices to 0-based indices by subtracting 1.
  else:
    r, taus = geqrf(a)

  if m > n and mode == 'left':
    zeros = jnp.zeros(c.shape[:-2] + (m - k,) + c.shape[-1:], dtype=c.dtype)
    c = jnp.concatenate([c, zeros], axis=-2)

  if conjugate:
    c = c.swapaxes(-1, -2)

  # conjugate swaps left/right because c and Q change sides when transposing.
  left = (mode == 'left') != conjugate
  cQ = ormqr(r, taus, c, left=left, transpose=conjugate)

  if conjugate:
    cQ = cQ.swapaxes(-1, -2)

  if mode == 'right':
    cQ = cQ[..., :k]

  if onedim:
    cQ = cQ.ravel()

  r = jnp.triu(r[..., :k, :])

  if pivoting:
    return cQ, r, p
  return cQ, r
