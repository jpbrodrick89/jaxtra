"""``jaxtra._src.lax.linalg`` — internal wrapper functions."""
from __future__ import annotations

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import ffi as _jax_ffi
from jax._src import core, dtypes
from jax._src.lax import lax
from jax._src.lax import control_flow
from jax._src.interpreters import mlir, ad, batching
from jax._src import dispatch
from jax._src.lax.linalg import (
    standard_linalg_primitive,
    register_cpu_gpu_lowering,
    register_module_custom_calls,
    _linalg_ffi_lowering,
    _float,
    _complex,
    _tril,
)
from jax._src.typing import ArrayLike, Array

from jaxtra._src.lib import lapack, gpu_solver, gpu_sparse


# ---------------------------------------------------------------------------
# Register FFI targets (mirrors jax._src.lax.linalg's pattern)
# ---------------------------------------------------------------------------

register_module_custom_calls(lapack)
register_module_custom_calls(gpu_solver)
register_module_custom_calls(gpu_sparse)


# ---------------------------------------------------------------------------
# Orthogonal QR multiply  (verbatim from PR #35104)
# ---------------------------------------------------------------------------

def ormqr(a: ArrayLike, taus: ArrayLike, c: ArrayLike, *,
          left: bool = True, transpose: bool = False) -> Array:
  """Multiplies a matrix by Q from a QR factorization without materializing Q.

  Computes ``Q @ C`` (``left=True``, ``transpose=False``),
  ``Q^T @ C`` (``left=True``, ``transpose=True``),
  ``C @ Q`` (``left=False``, ``transpose=False``), or
  ``C @ Q^T`` (``left=False``, ``transpose=True``).

  For complex types, ``transpose=True`` computes the conjugate transpose
  (``Q^H``).

  Args:
    a: The Householder reflectors from ``jax._src.lax.linalg.geqrf`` or
      ``jax._src.lax.linalg.geqp3``, with shape ``[..., m, n]``.
    taus: The Householder scalar factors from ``jax._src.lax.linalg.geqrf``
      or ``jax._src.lax.linalg.geqp3``, with shape ``[..., k]``.
    c: The matrix to multiply by Q, with shape ``[..., c_rows, c_cols]``.
    left: If ``True``, compute ``Q @ C``. If ``False``, compute ``C @ Q``.
    transpose: If ``True``, use ``Q^T`` (or ``Q^H`` for complex types).

  Returns:
    The result of multiplying ``c`` by Q (or ``Q^T``/``Q^H``), with the
    same shape as ``c``.
  """
  a, taus, c = core.standard_insert_pvary(a, taus, c)
  return ormqr_p.bind(a, taus, c, left=left, transpose=transpose)


def _ormqr_shape_rule(a_shape, taus_shape, c_shape, *, left, transpose):
  m = a_shape[0]
  if left and c_shape[0] != m:
    raise ValueError(
      "ormqr with left=True expects c to have the same number of rows as "
      f"the Householder matrix a. Got a shape {a_shape} and c shape {c_shape}.")
  if not left and c_shape[1] != m:
    raise ValueError(
      "ormqr with left=False expects c to have the same number of columns as "
      f"the Householder matrix a has rows. Got a shape {a_shape} and c shape {c_shape}.")
  return c_shape


def _ormqr_lowering(a, taus, c, *, left, transpose):
  # Apply Householder reflectors H_i = I - tau_i * v_i * v_i^H directly to c
  # without materializing Q. Cost: O(k * m * c_cols) if left,
  # O(k * c_rows * m) otherwise, where c has shape (..., c_rows, c_cols).
  *batch_dims, m, n = a.shape
  k = taus.shape[-1]
  is_complex = dtypes.issubdtype(a.dtype, np.complexfloating)

  # Householder vectors: lower triangle of a with unit diagonal.
  eye = lax._eye(a.dtype, (m, k))
  if batch_dims:
    eye = lax.broadcast(eye, tuple(batch_dims))
  V = _tril(a[..., :, :k], k=-1) + eye

  effective_taus = lax.conj(taus) if (transpose and is_complex) else taus

  # Q @ c and c @ Q^H apply reflectors in reverse; Q^H @ c and c @ Q forward.
  use_reverse = (left != transpose)

  n_batch = len(batch_dims)
  batch_contract = tuple(range(n_batch))

  def body(i, c):
    idx = (k - 1 - i) if use_reverse else i
    tau = effective_taus[..., idx]
    v = V[..., :, idx]
    tau_bc = lax.expand_dims(tau, (-1, -2))
    if left:
      # c = c - tau * v @ (v^H @ c)
      v_h = lax.conj(v) if is_complex else v
      vHc = lax.dot_general(v_h, c,
          (((v_h.ndim - 1,), (c.ndim - 2,)),
           (batch_contract, batch_contract)))
      update = lax.expand_dims(v, (-1,)) * lax.expand_dims(vHc, (-2,))
    else:
      # c = c - tau * (c @ v) @ v^H
      cv = lax.dot_general(c, v,
          (((c.ndim - 1,), (v.ndim - 1,)),
           (batch_contract, batch_contract)))
      v_h = lax.conj(v) if is_complex else v
      update = lax.expand_dims(cv, (-1,)) * lax.expand_dims(v_h, (-2,))
    return c - tau_bc * update

  return control_flow.fori_loop(0, k, body, c)


def _ormqr_cpu_gpu_lowering(ctx, a, taus, c, *, left, transpose,
                             target_name_prefix: str):
  a_aval, _, _ = ctx.avals_in
  if target_name_prefix == "cpu":
    dtype = a_aval.dtype
    prefix = "un" if dtypes.issubdtype(dtype, np.complexfloating) else "or"
    target_name = lapack.prepare_lapack_call(f"{prefix}mqr_ffi", dtype)
  else:
    target_name = f"{target_name_prefix}solver_ormqr_ffi"
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={2: 0})
  return rule(ctx, a, taus, c, left=left, transpose=transpose)


ormqr_p = standard_linalg_primitive(
    (_float | _complex, _float | _complex, _float | _complex), (2, 1, 2),
    _ormqr_shape_rule, "ormqr")
mlir.register_lowering(ormqr_p, mlir.lower_fun(
    _ormqr_lowering, multiple_results=False))
register_cpu_gpu_lowering(ormqr_p, _ormqr_cpu_gpu_lowering)


# ---------------------------------------------------------------------------
# Pentadiagonal solve
# ---------------------------------------------------------------------------
#
# Diagonal convention (matching cuSparse gpsvInterleavedBatch):
#   ds[i] = A[i, i-2]   (elements ds[0], ds[1] are unused / padding)
#   dl[i] = A[i, i-1]   (element dl[0] is unused / padding)
#   d[i]  = A[i, i]
#   du[i] = A[i, i+1]   (element du[n-1] is unused / padding)
#   dw[i] = A[i, i+2]   (elements dw[n-2], dw[n-1] are unused / padding)
#
# Solves A x = b, returning x with the same shape as b.


def pentadiagonal_solve(
    ds: ArrayLike, dl: ArrayLike, d: ArrayLike,
    du: ArrayLike, dw: ArrayLike, b: ArrayLike) -> Array:
  r"""Computes the solution of a pentadiagonal linear system.

  This function computes the solution of a pentadiagonal linear system:

  .. math::
    A \, X = B

  Args:
    ds: A batch of vectors with shape ``[..., m]``.
      The second lower diagonal of A: ``ds[i] := A[i, i-2]`` for i in ``[0, m)``.
      Note that ``ds[0]`` and ``ds[1]`` are unused.
    dl: A batch of vectors with shape ``[..., m]``.
      The lower diagonal of A: ``dl[i] := A[i, i-1]`` for i in ``[0, m)``.
      Note that ``dl[0] = 0``.
    d: A batch of vectors with shape ``[..., m]``.
      The middle diagonal of A: ``d[i] := A[i, i]`` for i in ``[0, m)``.
    du: A batch of vectors with shape ``[..., m]``.
      The upper diagonal of A: ``du[i] := A[i, i+1]`` for i in ``[0, m)``.
      Note that ``du[m - 1] = 0``.
    dw: A batch of vectors with shape ``[..., m]``.
      The second upper diagonal of A: ``dw[i] := A[i, i+2]`` for i in ``[0, m)``.
      Note that ``dw[m - 2]`` and ``dw[m - 1]`` are unused.
    b: Right hand side matrix.

  Returns:
    Solution ``X`` of pentadiagonal system.

  See also:
    - :func:`jax.lax.linalg.tridiagonal_solve`: Solves a tridiagonal system.
    - :func:`pentadiagonal_solveh`: Solves a Hermitian/SPD pentadiagonal system
      using banded Cholesky (faster when A is symmetric positive-definite).

  .. rubric:: Benchmarks

  SPD pentadiagonal systems (float64) compared against
  :func:`jax.numpy.linalg.solve` (dense LU),
  :func:`jax.scipy.linalg.cho_factor` / :func:`jax.scipy.linalg.cho_solve`
  (dense Cholesky), ``scipy.linalg.solve_banded``, and
  ``scipy.linalg.solveh_banded``.

  **CPU** — Both :func:`pentadiagonal_solve` and :func:`pentadiagonal_solveh`
  achieve O(kn) scaling vs O(n³) for dense solvers, and match or beat scipy's
  banded solvers while remaining fully ``jit`` / ``vmap`` / ``grad``
  compatible.

  .. figure:: /_bench_images/bench_banded.png
     :alt: CPU benchmark: pentadiagonal solve
     :width: 90%
     :align: center

  **GPU** — For a single system, GPU underperforms CPU for array sizes
  below ~10 000 (and potentially higher), likely due to the inherently
  sequential nature of banded solves limiting GPU utilisation. For batched
  solves the GPU is likely to outperform CPU thanks to cuSPARSE
  ``gpsvInterleavedBatch`` parallelism across independent systems. At large
  n the banded O(kn) advantage over dense solvers grows with system size.

  .. figure:: /_bench_images/bench_banded_gpu.png
     :alt: GPU benchmark: pentadiagonal solve
     :width: 90%
     :align: center

  See ``benchmarks/bench_banded.py`` for reproduction.
  """
  ds, dl, d, du, dw, b = core.standard_insert_pvary(ds, dl, d, du, dw, b)
  return pentadiagonal_solve_p.bind(ds, dl, d, du, dw, b)


def _pentadiagonal_solve_shape_rule(
    ds_shape, dl_shape, d_shape, du_shape, dw_shape, b_shape):
  if not (ds_shape == dl_shape == d_shape == du_shape == dw_shape):
    raise TypeError(
        "pentadiagonal_solve requires that all diagonal arguments have the "
        "same shape, got ds={}, dl={}, d={}, du={}, dw={}".format(
            ds_shape, dl_shape, d_shape, du_shape, dw_shape))
  if d_shape != b_shape[:-1]:
    raise TypeError(
        "pentadiagonal_solve requires that the leading ndim-1 dimensions of b "
        "equal the dimensions of the diagonal arguments.")
  return b_shape


def _pentadiagonal_solve_cpu_lowering(ctx, ds, dl, d, du, dw, b, **kwargs):
  del kwargs
  d_aval = ctx.avals_in[2]
  target_name = lapack.prepare_lapack_call("gbsv_ffi", d_aval.dtype)
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={5: 0})
  return rule(ctx, ds, dl, d, du, dw, b)


def _pentadiagonal_solve_gpu_lowering(ctx, ds, dl, d, du, dw, b, *,
                                       target_name_prefix):
  target_name = f"{target_name_prefix}sparse_gpsvInterleaved_ffi"
  # Identity layout (0, 1, ..., ndim-1) in minor-to-major places dim 0
  # innermost (stride 1).  For shape (batch, n, ...) this yields cuSPARSE's
  # interleaved-batch format: data[i * batchCount + b].
  operand_layouts = [tuple(range(len(aval.shape)))
                     for aval in ctx.avals_in]
  result_layouts = [tuple(range(len(aval.shape)))
                    for aval in ctx.avals_out]
  rule = _jax_ffi.ffi_lowering(
      target_name,
      operand_layouts=operand_layouts,
      result_layouts=result_layouts,
      operand_output_aliases={5: 0})
  return rule(ctx, ds, dl, d, du, dw, b)


def _pentadiagonal_product(ds, dl, d, du, dw, v):
  """A @ v where A is pentadiagonal and v is (..., n, k)."""
  y = lax.reshape(d, d.shape + (1,)) * v
  y = y.at[..., 1:, :].add(dl[..., 1:, None] * v[..., :-1, :])
  y = y.at[..., 2:, :].add(ds[..., 2:, None] * v[..., :-2, :])
  y = y.at[..., :-1, :].add(du[..., :-1, None] * v[..., 1:, :])
  y = y.at[..., :-2, :].add(dw[..., :-2, None] * v[..., 2:, :])
  return y


def _pentadiagonal_solve_jvp(primals, tangents):
  ds, dl, d, du, dw, b = primals
  dds, ddl, dd, ddu, ddw, db = tangents

  x = pentadiagonal_solve_p.bind(ds, dl, d, du, dw, b)

  # Replace Zero tangents with actual zeros.
  z = lambda t, ref: jnp.zeros_like(ref) if type(t) is ad.Zero else t
  dds = z(dds, ds); ddl = z(ddl, dl); dd = z(dd, d)
  ddu = z(ddu, du); ddw = z(ddw, dw); db = z(db, b)

  # Compute dA @ x (matvec of the perturbation matrix applied to x).
  dAx = _pentadiagonal_product(dds, ddl, dd, ddu, ddw, x)

  # dx = A^{-1} (db - dA x)
  dx = pentadiagonal_solve_p.bind(ds, dl, d, du, dw, db - dAx)
  return x, dx


def _pentadiagonal_solve_transpose(ct, ds, dl, d, du, dw, b):
  # ct is the cotangent of x = A^{-1} b.
  # We compute the cotangent of b: ct_b = A^{-T} ct.
  #
  # A^T diagonal relationship:
  #   (A^T)[i, i-2] = A[i-2, i] = dw[i-2]  (lower-2 of A^T)
  #   (A^T)[i, i-1] = A[i-1, i] = du[i-1]  (lower-1 of A^T)
  #   (A^T)[i, i]   = A[i,   i] = d[i]      (main of A^T)
  #   (A^T)[i, i+1] = A[i+1, i] = dl[i+1]  (upper-1 of A^T)
  #   (A^T)[i, i+2] = A[i+2, i] = ds[i+2]  (upper-2 of A^T)
  zeros_1 = lax.full_like(d, 0, shape=d.shape[:-1] + (1,))
  zeros_2 = lax.full_like(d, 0, shape=d.shape[:-1] + (2,))

  # Build A^T diagonals.
  ds_T = lax.concatenate([zeros_2, dw[..., :-2]], dimension=len(d.shape) - 1)
  dl_T = lax.concatenate([zeros_1, du[..., :-1]], dimension=len(d.shape) - 1)
  d_T = d
  du_T = lax.concatenate([dl[..., 1:], zeros_1], dimension=len(d.shape) - 1)
  dw_T = lax.concatenate([ds[..., 2:], zeros_2], dimension=len(d.shape) - 1)

  ct_b = pentadiagonal_solve_p.bind(ds_T, dl_T, d_T, du_T, dw_T, ct)
  return [None, None, None, None, None, ct_b]


def _pentadiagonal_solve_jax_fallback(ds, dl, d, du, dw, b):
  """Pure-JAX fallback: reconstruct dense matrix and use LU solve."""
  n = d.shape[-1]
  # Build full n x n matrix from the five diagonals.
  A = jnp.diag(d)
  A = A.at[jnp.arange(1, n), jnp.arange(n - 1)].add(dl[1:])
  A = A.at[jnp.arange(2, n), jnp.arange(n - 2)].add(ds[2:])
  A = A.at[jnp.arange(n - 1), jnp.arange(1, n)].add(du[:n - 1])
  A = A.at[jnp.arange(n - 2), jnp.arange(2, n)].add(dw[:n - 2])
  # b has shape (n, k); jnp.linalg.solve handles matrix RHS natively.
  return jnp.linalg.solve(A, b)


pentadiagonal_solve_p = standard_linalg_primitive(
    (_float | _complex,) * 6, (1, 1, 1, 1, 1, 2),
    _pentadiagonal_solve_shape_rule, "pentadiagonal_solve")

# Register JVP and transpose rules.
ad.primitive_jvps[pentadiagonal_solve_p] = _pentadiagonal_solve_jvp
ad.primitive_transposes[pentadiagonal_solve_p] = _pentadiagonal_solve_transpose

# Register pure-JAX fallback lowering (used on TPU / platforms without FFI).
mlir.register_lowering(pentadiagonal_solve_p, mlir.lower_fun(
    _pentadiagonal_solve_jax_fallback, multiple_results=False))

# Register CPU (LAPACK gbsv) and GPU (cuSPARSE gpsvInterleavedBatch) lowerings.
mlir.register_lowering(pentadiagonal_solve_p,
                        _pentadiagonal_solve_cpu_lowering, platform='cpu')
mlir.register_lowering(pentadiagonal_solve_p,
                        partial(_pentadiagonal_solve_gpu_lowering,
                                target_name_prefix='cu'),
                        platform='cuda')


def _pentadiagonal_solve_batching_rule(batched_args, batch_dims):
  ds, dl, d, du, dw, b = batched_args
  bds, bdl, bd, bdu, bdw, bb = batch_dims
  if all(bd is batching.not_mapped for bd in (bds, bdl, bd, bdu, bdw)):
    b = batching.moveaxis(b, bb, -2)
    b_flat = b.reshape(b.shape[:-3] + (b.shape[-3], b.shape[-2] * b.shape[-1]))
    bdim_out = b.ndim - 2
    out_flat = pentadiagonal_solve(ds, dl, d, du, dw, b_flat)
    return out_flat.reshape(b.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    ds = batching.bdim_at_front(ds, bds, size)
    dl = batching.bdim_at_front(dl, bdl, size)
    d  = batching.bdim_at_front(d,  bd,  size)
    du = batching.bdim_at_front(du, bdu, size)
    dw = batching.bdim_at_front(dw, bdw, size)
    b  = batching.bdim_at_front(b,  bb,  size)
    return pentadiagonal_solve(ds, dl, d, du, dw, b), 0

batching.primitive_batchers[pentadiagonal_solve_p] = (
    _pentadiagonal_solve_batching_rule)


# ---------------------------------------------------------------------------
# Hermitian pentadiagonal solve
# ---------------------------------------------------------------------------
#
# Upper-triangle convention (matching LAPACK pbsv UPLO='U'):
#   d[i]  = A[i, i]
#   du[i] = A[i, i+1]   (element du[n-1] is unused / padding)
#   dw[i] = A[i, i+2]   (elements dw[n-2], dw[n-1] are unused / padding)
#
# Lower triangle is implied: A[i, i-1] = conj(du[i-1]), A[i, i-2] = conj(dw[i-2]).
#
# Solves A x = b, returning x with the same shape as b.


def _hermitian_lower_diags(du, dw):
  """Reconstruct lower diagonals from upper diagonals of a Hermitian matrix."""
  is_complex = dtypes.issubdtype(du.dtype, np.complexfloating)
  conj = lax.conj if is_complex else lambda x: x
  z1 = lax.full_like(du, 0, shape=du.shape[:-1] + (1,))
  z2 = lax.full_like(dw, 0, shape=dw.shape[:-1] + (2,))
  dl = lax.concatenate([z1, conj(du[..., :-1])],
                       dimension=len(du.shape) - 1)
  ds = lax.concatenate([z2, conj(dw[..., :-2])],
                       dimension=len(dw.shape) - 1)
  return ds, dl


def pentadiagonal_solveh(
    d: ArrayLike, du: ArrayLike, dw: ArrayLike,
    b: ArrayLike) -> Array:
  r"""Computes the solution of a Hermitian pentadiagonal linear system.

  This function computes the solution of a Hermitian (symmetric for real
  types) pentadiagonal linear system:

  .. math::
    A \, X = B

  Only the upper-triangle diagonals are stored; the lower triangle is implied
  by symmetry: ``A[i, i-1] = conj(du[i-1])``, ``A[i, i-2] = conj(dw[i-2])``.

  Args:
    d: A batch of vectors with shape ``[..., m]``.
      The middle diagonal of A: ``d[i] := A[i, i]`` for i in ``[0, m)``.
    du: A batch of vectors with shape ``[..., m]``.
      The upper diagonal of A: ``du[i] := A[i, i+1]`` for i in ``[0, m)``.
      Note that ``du[m - 1] = 0``.
    dw: A batch of vectors with shape ``[..., m]``.
      The second upper diagonal of A: ``dw[i] := A[i, i+2]`` for i in ``[0, m)``.
      Note that ``dw[m - 2]`` and ``dw[m - 1]`` are unused.
    b: Right hand side matrix.

  Returns:
    Solution ``X`` of Hermitian pentadiagonal system.

  Note:
    On GPU (and other platforms without a dedicated FFI target), this falls
    back to :func:`pentadiagonal_solve` by reconstructing the lower diagonals
    from symmetry. The dedicated LAPACK ``pbsv`` (banded Cholesky) kernel is
    only used on CPU.

  See also:
    - :func:`jax.lax.linalg.tridiagonal_solve`: Solves a tridiagonal system.
    - :func:`pentadiagonal_solve`: Solves a general (non-symmetric)
      pentadiagonal system.

  .. rubric:: Benchmarks

  SPD pentadiagonal systems (float64) compared against
  :func:`jax.numpy.linalg.solve` (dense LU),
  :func:`jax.scipy.linalg.cho_factor` / :func:`jax.scipy.linalg.cho_solve`
  (dense Cholesky), ``scipy.linalg.solve_banded``, and
  ``scipy.linalg.solveh_banded``.

  **CPU** — Both :func:`pentadiagonal_solve` and :func:`pentadiagonal_solveh`
  achieve O(kn) scaling vs O(n³) for dense solvers, and match or beat scipy's
  banded solvers while remaining fully ``jit`` / ``vmap`` / ``grad``
  compatible.

  .. figure:: /_bench_images/bench_banded.png
     :alt: CPU benchmark: pentadiagonal solve
     :width: 90%
     :align: center

  **GPU** — For a single system, GPU underperforms CPU for array sizes
  below ~10 000 (and potentially higher), likely due to the inherently
  sequential nature of banded solves limiting GPU utilisation. For batched
  solves the GPU is likely to outperform CPU thanks to cuSPARSE
  ``gpsvInterleavedBatch`` parallelism across independent systems. At large
  n the banded O(kn) advantage over dense solvers grows with system size.

  .. figure:: /_bench_images/bench_banded_gpu.png
     :alt: GPU benchmark: pentadiagonal solve
     :width: 90%
     :align: center

  See ``benchmarks/bench_banded.py`` for reproduction.
  """
  d, du, dw, b = core.standard_insert_pvary(d, du, dw, b)
  return pentadiagonal_solveh_p.bind(d, du, dw, b)


def _pentadiagonal_solveh_shape_rule(d_shape, du_shape, dw_shape, b_shape):
  if not (d_shape == du_shape == dw_shape):
    raise TypeError(
        "pentadiagonal_solveh requires that all diagonal arguments have the "
        "same shape, got d={}, du={}, dw={}".format(d_shape, du_shape, dw_shape))
  if d_shape != b_shape[:-1]:
    raise TypeError(
        "pentadiagonal_solveh requires that the leading ndim-1 dimensions of b "
        "equal the dimensions of the diagonal arguments.")
  return b_shape


def _pentadiagonal_solveh_cpu_lowering(ctx, d, du, dw, b, **kwargs):
  del kwargs
  d_aval = ctx.avals_in[0]
  target_name = lapack.prepare_lapack_call("pbsv_ffi", d_aval.dtype)
  rule = _linalg_ffi_lowering(target_name, operand_output_aliases={3: 0})
  return rule(ctx, d, du, dw, b)


def _pentadiagonal_solveh_fallback(d, du, dw, b):
  """Reconstruct full pentadiagonal and delegate to pentadiagonal_solve."""
  ds, dl = _hermitian_lower_diags(du, dw)
  return pentadiagonal_solve(ds, dl, d, du, dw, b)


def _pentadiagonal_solveh_jvp(primals, tangents):
  d, du, dw, b = primals
  dd, ddu, ddw, db = tangents

  x = pentadiagonal_solveh_p.bind(d, du, dw, b)

  # Replace Zero tangents with actual zeros.
  z = lambda t, ref: jnp.zeros_like(ref) if type(t) is ad.Zero else t
  dd = z(dd, d); ddu = z(ddu, du); ddw = z(ddw, dw); db = z(db, b)

  # Build lower-triangle tangent diagonals from upper tangents.
  dds, ddl = _hermitian_lower_diags(ddu, ddw)

  # dA @ x
  dAx = _pentadiagonal_product(dds, ddl, dd, ddu, ddw, x)

  # dx = A^{-1} (db - dA x)
  dx = pentadiagonal_solveh_p.bind(d, du, dw, db - dAx)
  return x, dx


def _pentadiagonal_solveh_transpose(ct, d, du, dw, b):
  # ct is the cotangent of x = A^{-1} b.
  # We need ct_b = A^{-T} ct.
  # For real symmetric: A^T = A, so ct_b = A^{-1} ct.
  # For complex Hermitian: A^T = conj(A), so A^{-T} = conj(A)^{-1}.
  # conj(A) is Hermitian with upper diags conj(d), conj(du), conj(dw).
  is_complex = dtypes.issubdtype(d.dtype, np.complexfloating)
  if is_complex:
    ct_b = pentadiagonal_solveh_p.bind(
        lax.conj(d), lax.conj(du), lax.conj(dw), ct)
  else:
    ct_b = pentadiagonal_solveh_p.bind(d, du, dw, ct)
  return [None, None, None, ct_b]


pentadiagonal_solveh_p = standard_linalg_primitive(
    (_float | _complex,) * 4, (1, 1, 1, 2),
    _pentadiagonal_solveh_shape_rule, "pentadiagonal_solveh")

# Register JVP and transpose rules.
ad.primitive_jvps[pentadiagonal_solveh_p] = _pentadiagonal_solveh_jvp
ad.primitive_transposes[pentadiagonal_solveh_p] = (
    _pentadiagonal_solveh_transpose)

# Register fallback lowering (reconstructs lower diags, calls pentadiagonal_solve).
mlir.register_lowering(pentadiagonal_solveh_p, mlir.lower_fun(
    _pentadiagonal_solveh_fallback, multiple_results=False))

# Register CPU lowering (LAPACK pbsv — banded Cholesky, faster than gbsv).
mlir.register_lowering(pentadiagonal_solveh_p,
                        _pentadiagonal_solveh_cpu_lowering, platform='cpu')


def _pentadiagonal_solveh_batching_rule(batched_args, batch_dims):
  d, du, dw, b = batched_args
  bd, bdu, bdw, bb = batch_dims
  if all(dim is batching.not_mapped for dim in (bd, bdu, bdw)):
    b = batching.moveaxis(b, bb, -2)
    b_flat = b.reshape(
        b.shape[:-3] + (b.shape[-3], b.shape[-2] * b.shape[-1]))
    bdim_out = b.ndim - 2
    out_flat = pentadiagonal_solveh(d, du, dw, b_flat)
    return out_flat.reshape(b.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    d  = batching.bdim_at_front(d,  bd,  size)
    du = batching.bdim_at_front(du, bdu, size)
    dw = batching.bdim_at_front(dw, bdw, size)
    b  = batching.bdim_at_front(b,  bb,  size)
    return pentadiagonal_solveh(d, du, dw, b), 0

batching.primitive_batchers[pentadiagonal_solveh_p] = (
    _pentadiagonal_solveh_batching_rule)


# ---------------------------------------------------------------------------
# Tridiagonal LU factorization  (LAPACK gttrf)
# ---------------------------------------------------------------------------
#
# Input convention (mirrors JAX's tridiagonal_solve padding):
#   dl[..., i] = A[i, i-1]  for i in [0, n):  dl[..., 0] is padding
#   d[..., i]  = A[i, i]    for i in [0, n)
#   du[..., i] = A[i, i+1]  for i in [0, n):  du[..., n-1] is padding
#
# All three diagonals have shape (..., n).
#
# Returns (dl_out, d_out, du_out, du2_out, ipiv_out) where:
#   dl_out  : (..., n) — L multipliers; position 0 is padding
#   d_out   : (..., n) — U main diagonal
#   du_out  : (..., n) — U first superdiagonal; position n-1 is padding
#   du2_out : (..., n) — U second superdiagonal; last two elements are 0
#   ipiv_out: (..., n) int32 — 1-based pivot indices


def gttrf(dl: ArrayLike, d: ArrayLike, du: ArrayLike
          ) -> tuple[Array, Array, Array, Array, Array]:
  """Computes the LU factorization of a tridiagonal matrix.

  Wraps LAPACK ``gttrf``.  All three diagonals must be padded to length ``n``
  (the main-diagonal length), following JAX's ``tridiagonal_solve`` convention:
  ``dl[..., 0]`` and ``du[..., n-1]`` are unused padding.

  Args:
    dl: Subdiagonal, shape ``(..., n)``.  ``dl[..., 0]`` is ignored.
    d:  Main diagonal, shape ``(..., n)``.
    du: Superdiagonal, shape ``(..., n)``.  ``du[..., n-1]`` is ignored.

  Returns:
    A five-tuple ``(dl_out, d_out, du_out, du2_out, ipiv_out)``:

    * **dl_out** ``(..., n)`` — lower-bidiagonal factor L (unit diagonal
      implied); ``dl_out[..., 0]`` is padding.
    * **d_out** ``(..., n)`` — main diagonal of U.
    * **du_out** ``(..., n)`` — first superdiagonal of U;
      ``du_out[..., n-1]`` is padding.
    * **du2_out** ``(..., n)`` — second superdiagonal of U (arises from row
      pivoting); ``du2_out[..., n-2]`` and ``du2_out[..., n-1]`` are zero.
    * **ipiv_out** ``(..., n)`` int32 — 1-based pivot indices.
      Row ``i`` was interchanged with row ``ipiv_out[..., i] - 1``.
  """
  dl, d, du = core.standard_insert_pvary(dl, d, du)
  return gttrf_p.bind(dl, d, du)


def _gttrf_abstract_eval(dl_aval, d_aval, du_aval):
  return (
      dl_aval,
      d_aval,
      du_aval,
      d_aval,  # du2_out: same float shape as d
      d_aval.update(dtype=np.dtype('int32')),  # ipiv_out
  )


def _gttrf_fallback(dl, d, du):
  """Pure-JAX fallback: Thomas algorithm (no pivoting) for non-CPU platforms."""
  n = d.shape[-1]

  def step(carry, i):
    dl_f, d_f, du2_f = carry
    mult = lax.dynamic_index_in_dim(dl_f, i + 1, axis=-1, keepdims=False) / \
           lax.dynamic_index_in_dim(d_f,  i,     axis=-1, keepdims=False)
    dl_f  = lax.dynamic_update_index_in_dim(dl_f, mult, i + 1, axis=-1)
    new_d = lax.dynamic_index_in_dim(d_f,  i + 1, axis=-1, keepdims=False) - \
            mult * lax.dynamic_index_in_dim(du, i, axis=-1, keepdims=False)
    d_f   = lax.dynamic_update_index_in_dim(d_f, new_d, i + 1, axis=-1)
    return (dl_f, d_f, du2_f), None

  du2_init = jnp.zeros_like(d)
  (dl_out, d_out, du2_out), _ = lax.scan(
      step, (dl, d, du2_init), jnp.arange(n - 1))
  ipiv_out = jnp.broadcast_to(
      jnp.arange(1, n + 1, dtype=jnp.int32), d.shape)
  return dl_out, d_out, du, du2_out, ipiv_out


def _gttrf_cpu_lowering(ctx, dl, d, du, **kwargs):
  del kwargs
  d_aval = ctx.avals_in[1]
  target_name = lapack.prepare_lapack_call("gttrf_ffi", d_aval.dtype)
  # Use ffi_lowering directly: our inputs are rank-1 (1D diagonals), so
  # _linalg_ffi_lowering's sharding rule (which assumes rank-2) would fail.
  rule = _jax_ffi.ffi_lowering(target_name, operand_output_aliases={0: 0, 1: 1, 2: 2})
  return rule(ctx, dl, d, du)


gttrf_p = core.Primitive("gttrf")
gttrf_p.multiple_results = True
gttrf_p.def_abstract_eval(_gttrf_abstract_eval)
gttrf_p.def_impl(partial(dispatch.apply_primitive, gttrf_p))

mlir.register_lowering(gttrf_p, mlir.lower_fun(
    _gttrf_fallback, multiple_results=True))
mlir.register_lowering(gttrf_p, _gttrf_cpu_lowering, platform='cpu')


def _gttrf_batching_rule(batched_args, batch_dims):
  dl, d, du = batched_args
  bdl, bd, bdu = batch_dims
  size = next(a.shape[i] for a, i in zip([dl, d, du], [bdl, bd, bdu])
              if i is not None)
  dl = batching.bdim_at_front(dl, bdl, size)
  d  = batching.bdim_at_front(d,  bd,  size)
  du = batching.bdim_at_front(du, bdu, size)
  results = gttrf_p.bind(dl, d, du)
  return results, (0,) * len(results)

batching.primitive_batchers[gttrf_p] = _gttrf_batching_rule


# ---------------------------------------------------------------------------
# Determinant helpers built on gttrf
# ---------------------------------------------------------------------------

def tridiag_logdet_lu(
    a: ArrayLike, b: ArrayLike, c: ArrayLike
) -> tuple[Array, Array]:
  """Log-determinant of a tridiagonal matrix via LAPACK gttrf.

  Computes ``(sign, log|det(A)|)`` where A is the tridiagonal matrix with
  subdiagonal ``a``, main diagonal ``b``, and superdiagonal ``c``.

  Input convention matches the user-facing scan-based functions:

    ``a[i]`` = A[i+1, i]  for i in [0, n-2]   — subdiagonal  (n-1,)
    ``b[i]`` = A[i, i]    for i in [0, n-1]   — main diagonal (n,)
    ``c[i]`` = A[i, i+1]  for i in [0, n-2]   — superdiagonal (n-1,)

  Returns:
    A two-tuple ``(sign, logabsdet)`` of scalars.
  """
  a = jnp.asarray(a)
  b = jnp.asarray(b)
  c = jnp.asarray(c)
  n = b.shape[-1]
  # Pad a and c to length n (JAX tridiagonal convention).
  zeros = jnp.zeros(a.shape[:-1] + (1,), dtype=a.dtype)
  dl = jnp.concatenate([zeros, a], axis=-1)   # dl[..., 0] = 0 (padding)
  du = jnp.concatenate([c, zeros], axis=-1)   # du[..., n-1] = 0 (padding)
  _, d_out, _, _, ipiv_out = gttrf(dl, b, du)
  # det(A) = prod(U diagonal) * (-1)^(number of row swaps)
  logabsdet = jnp.sum(jnp.log(jnp.abs(d_out)), axis=-1)
  swaps = jnp.sum(
      ipiv_out != jnp.arange(1, n + 1, dtype=jnp.int32), axis=-1)
  sign = jnp.prod(jnp.sign(d_out), axis=-1) * ((-1.0) ** swaps)
  return sign, logabsdet


tridiag_logdet_lu_jit = jax.jit(tridiag_logdet_lu)
tridiag_logdet_lu_batched = jax.jit(jax.vmap(tridiag_logdet_lu))
