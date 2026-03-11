"""``jaxtra.scipy.linalg`` – mirrors the interface of ``jax.scipy.linalg``.

Currently exposes:

* :func:`qr_multiply` – combined QR decomposition + Q-multiply step.
* :func:`ldl` – LDL factorization for symmetric/Hermitian indefinite matrices.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
import jax.numpy as jnp

from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike
from jax._src.lax.linalg import geqrf, geqp3

from jaxtra._src.lax.linalg import (
  ormqr,
  ldl as _ldl_primitive,
  ldl_solve,
  _reconstruct_ldl_numpy,
)

__all__ = ["qr_multiply", "ldl", "ldl_solve"]


@overload
def qr_multiply(
  a: ArrayLike,
  c: ArrayLike,
  mode: str = "right",
  pivoting: Literal[False] = False,
  conjugate: bool = False,
  overwrite_a: bool = False,
  overwrite_c: bool = False,
) -> tuple[Array, Array]: ...


@overload
def qr_multiply(
  a: ArrayLike,
  c: ArrayLike,
  mode: str = "right",
  pivoting: Literal[True] = True,
  conjugate: bool = False,
  overwrite_a: bool = False,
  overwrite_c: bool = False,
) -> tuple[Array, Array, Array]: ...


@overload
def qr_multiply(
  a: ArrayLike,
  c: ArrayLike,
  mode: str = "right",
  pivoting: bool = False,
  conjugate: bool = False,
  overwrite_a: bool = False,
  overwrite_c: bool = False,
) -> tuple[Array, Array] | tuple[Array, Array, Array]: ...


def qr_multiply(
  a: ArrayLike,
  c: ArrayLike,
  mode: str = "right",
  pivoting: bool = False,
  conjugate: bool = False,
  overwrite_a: bool = False,
  overwrite_c: bool = False,
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
    - :func:`jaxtra._src.lax.linalg.ormqr`: XLA-style Q-multiply primitive

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

  .. rubric:: Benchmarks

  Least-squares solve **A x ≈ b** (float64) compared against
  :func:`jax.numpy.linalg.qr` (dense QR, Q materialised),
  :func:`jax.numpy.linalg.lstsq` (SVD-based), and
  :obj:`scipy.linalg.lapack.dgels` (LAPACK QR, CPU only).

  **CPU** — On tall, skinny systems (M ≫ N, N ≈ 50–200), :func:`qr_multiply`
  is **1.7–2.5×** faster than dense QR and **2–3.4×** faster than
  :func:`~jax.numpy.linalg.lstsq` at 100 columns / 5 000–20 000 rows.
  Speedups grow with the number of columns.

  **GPU** — For column counts below ~200, :func:`qr_multiply` can be
  *slower* than dense QR because cuSOLVER's ``orgqr`` + highly optimised
  cuBLAS GEMM/GEMV kernels are hard to beat. At higher column counts
  (200–512) avoiding Q materialisation helps, but speedups are typically
  <10% for the same reason.

  See ``benchmarks/bench_least_squares.py`` for reproduction.

  .. raw:: html

     <details><summary>CPU benchmarks</summary>

  .. figure:: /_bench_images/bench_cols20.png
     :alt: CPU benchmark: 20 columns
     :width: 90%
     :align: center

  .. figure:: /_bench_images/bench_cols50.png
     :alt: CPU benchmark: 50 columns
     :width: 90%
     :align: center

  .. figure:: /_bench_images/bench_cols100.png
     :alt: CPU benchmark: 100 columns
     :width: 90%
     :align: center

  .. raw:: html

     </details>
     <details><summary>GPU benchmarks</summary>

  .. figure:: /_bench_images/bench_cols20_gpu.png
     :alt: GPU benchmark: 20 columns
     :width: 90%
     :align: center

  .. figure:: /_bench_images/bench_cols50_gpu.png
     :alt: GPU benchmark: 50 columns
     :width: 90%
     :align: center

  .. figure:: /_bench_images/bench_cols100_gpu.png
     :alt: GPU benchmark: 100 columns
     :width: 90%
     :align: center

  .. figure:: /_bench_images/bench_cols200_gpu.png
     :alt: GPU benchmark: 200 columns
     :width: 90%
     :align: center

  .. figure:: /_bench_images/bench_cols512_gpu.png
     :alt: GPU benchmark: 512 columns
     :width: 90%
     :align: center

  .. raw:: html

     </details>
  """
  del overwrite_a, overwrite_c  # unused
  a, c = promote_dtypes_inexact(jnp.asarray(a), jnp.asarray(c))
  if mode not in ("right", "left"):
    raise ValueError(f"mode must be 'right' or 'left', got {mode!r}")

  onedim = c.ndim == 1
  if onedim:
    c = c[:, None] if mode == "left" else c[None, :]

  m, n = a.shape[-2:]
  k = min(m, n)

  if mode == "left":
    if c.shape[-2] != k:
      raise ValueError(
        f"Array shapes are not compatible for Q @ c operation: "
        f"a has shape {tuple(a.shape)} so Q has {k} columns, "
        f"but c has {c.shape[-2]} rows (expected {k})."
      )
  else:
    if c.shape[-1] != m:
      raise ValueError(
        f"Array shapes are not compatible for c @ Q operation: "
        f"a has shape {tuple(a.shape)} so Q has {m} rows, "
        f"but c has {c.shape[-1]} columns (expected {m})."
      )

  if pivoting:
    jpvt = jnp.zeros(a.shape[:-2] + (n,), dtype=jnp.int32)
    r, p, taus = geqp3(a, jpvt)
    p -= (
      1  # Convert geqp3's 1-based indices to 0-based indices by subtracting 1.
    )
  else:
    r, taus = geqrf(a)

  if m > n and mode == "left":
    zeros = jnp.zeros(c.shape[:-2] + (m - k,) + c.shape[-1:], dtype=c.dtype)
    c = jnp.concatenate([c, zeros], axis=-2)

  if conjugate:
    c = c.swapaxes(-1, -2)

  # conjugate swaps left/right because c and Q change sides when transposing.
  left = (mode == "left") != conjugate
  cQ = ormqr(r, taus, c, left=left, transpose=conjugate)

  if conjugate:
    cQ = cQ.swapaxes(-1, -2)

  if mode == "right":
    cQ = cQ[..., :k]

  if onedim:
    cQ = cQ.ravel()

  r = jnp.triu(r[..., :k, :])

  if pivoting:
    return cQ, r, p
  return cQ, r


def ldl(
  a: ArrayLike,
  lower: bool = True,
  hermitian: bool | None = None,
  overwrite_a: bool = False,
  check_finite: bool = True,
) -> tuple[Array, Array, Array]:
  """LDL factorization of a symmetric or Hermitian indefinite matrix.

  JAX implementation of :func:`scipy.linalg.ldl`.  Uses LAPACK ``?sytrf``
  (symmetric) or ``?hetrf`` (Hermitian) on CPU via jaxtra's custom FFI
  binding, falling back to LU decomposition on platforms without the binding.

  Args:
    a: Square matrix of shape ``(..., n, n)``.  Must be symmetric (when
      ``hermitian=False``) or Hermitian (when ``hermitian=True``).
    lower: If ``True`` (default), factor the lower triangle.  The result is
      ``P @ A @ P^T = L @ D @ L^{T/H}``.  If ``False``, factor the upper
      triangle: ``P @ A @ P^T = U^{T/H} @ D @ U``.
    hermitian: If ``None`` (default), treated as ``True`` when ``a`` has a
      complex dtype and ``False`` otherwise.  Set explicitly to override.
    overwrite_a: Unused in JAX (kept for scipy compatibility).
    check_finite: Unused in JAX (kept for scipy compatibility).

  Returns:
    A tuple ``(lu, d, perm)`` where:

    * ``lu`` — lower (``lower=True``) or upper (``lower=False``) triangular
      factor with unit diagonal.  Shape ``(..., n, n)``.
    * ``d`` — block-diagonal D factor.  Shape ``(..., n, n)``.
    * ``perm`` — 0-indexed permutation array of shape ``(..., n)`` such that
      ``lu @ d @ lu.conj().T == a[..., perm, :][..., :, perm]``
      (for Hermitian) or with ``lu.T`` (for symmetric).

  See Also:
    :func:`jaxtra.scipy.linalg.ldl_solve`: Solve ``A @ x = b`` using the raw LDL factorization (JIT-compatible).
    :func:`jaxtra._src.lax.linalg.ldl`: Raw primitive returning ``(factors, ipiv)`` directly (JIT-compatible).
    :func:`jax.scipy.linalg.lu_factor`: Analogous LU factorization.
    :func:`jax.scipy.linalg.cho_factor`: Analogous Cholesky factorization (for positive-definite matrices).

  Examples:
    Factorize and solve a symmetric indefinite system ``A @ x = b``:

    >>> import jax.numpy as jnp
    >>> import jaxtra.scipy.linalg as jsl
    >>> from jaxtra._src.lax.linalg import ldl as ldl_prim, ldl_solve
    >>> A = jnp.array([[2., 1.], [1., -3.]])
    >>> lu, d, perm = jsl.ldl(A)
    >>> # Verify: lu @ d @ lu.T == A[perm][:, perm]
    >>> # For a JIT-compatible solve, use the primitive:
    >>> factors, ipiv = ldl_prim(A)
    >>> x = ldl_solve(factors, ipiv, jnp.array([1., 2.]))
  """
  del overwrite_a, check_finite  # unused
  a = jnp.asarray(a)
  if hermitian is None:
    hermitian = jnp.issubdtype(a.dtype, jnp.complexfloating)
  hermitian = bool(hermitian)

  # Promote to inexact dtype.
  (a,) = promote_dtypes_inexact(a)

  if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
    raise ValueError(f"ldl requires a square matrix, got shape {a.shape}")

  # Call the primitive (JIT-compatible).
  factors, ipiv = _ldl_primitive(a, lower=lower, hermitian=hermitian)

  # Reconstruct (lu, d, perm) using numpy (not JIT-traceable, like scipy).
  factors_np = np.array(factors)
  ipiv_np = np.array(ipiv)

  batch_shape = a.shape[:-2]
  n = a.shape[-1]

  if batch_shape:
    factors_flat = factors_np.reshape(-1, n, n)
    ipiv_flat = ipiv_np.reshape(-1, n)

    lu_list, d_list, perm_list = [], [], []
    for i in range(factors_flat.shape[0]):
      lu_i, d_i, perm_i = _reconstruct_ldl_numpy(
        factors_flat[i], ipiv_flat[i], lower=lower, hermitian=hermitian
      )
      lu_list.append(lu_i)
      d_list.append(d_i)
      perm_list.append(perm_i)

    import numpy as _np

    lu_np = _np.stack(lu_list).reshape(batch_shape + (n, n))
    d_np = _np.stack(d_list).reshape(batch_shape + (n, n))
    perm_np = _np.stack(perm_list).reshape(batch_shape + (n,))
  else:
    lu_np, d_np, perm_np = _reconstruct_ldl_numpy(
      factors_np, ipiv_np, lower=lower, hermitian=hermitian
    )

  return jnp.array(lu_np), jnp.array(d_np), jnp.array(perm_np, dtype=jnp.int32)
