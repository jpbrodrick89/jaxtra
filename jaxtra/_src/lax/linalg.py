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
from jax._src.lax.linalg import (
    standard_linalg_primitive,
    linalg_primitive,
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
# Triangular-pentagonal QR (LAPACK tpqrt)
# ---------------------------------------------------------------------------
#
# Computes the QR factorization of a "triangular-pentagonal" matrix
#
#       C = [ A ]      A : (n, n) upper triangular
#           [ B ]      B : (m, n) pentagonal
#
# and returns only the re-triangularised factor R = triu(qr(C)), with shape
# (n, n).  The pentagonal B is an (m - l)-by-n rectangular block on top of an
# l-by-n upper trapezoidal block (B[m - l + i, j] = 0 for j < i).
#
# This is the structured QR update of a trust-region Levenberg-Marquardt step:
# A is the triangular factor R of the Jacobian (from ``geqrf``) and B is the
# diagonal regularisation D (m = n, l = n).  Re-triangularising [R; D] yields
# the factor R̃ with R̃ᴴR̃ = RᴴR + DᴴD without forming the normal equations.


def tpqrt(a: ArrayLike, b: ArrayLike, *,
          l: int | None = None, nb: int | None = None
          ) -> tuple[Array, Array, Array]:
  r"""Blocked QR factorization of a triangular-pentagonal matrix.

  Computes the QR factorization of ``C = [a; b]`` where ``a`` is an ``(n, n)``
  upper triangular matrix stacked on top of an ``(m, n)`` pentagonal matrix
  ``b``, so that ``C = Q [R; 0]``. Mirrors :obj:`scipy.linalg.lapack.dtpqrt`
  (inputs ``l, nb, a, b``; outputs ``R, V, T``).

  The canonical use is the trust-region Levenberg-Marquardt update: given the
  triangular factor ``R`` of a Jacobian and a diagonal regularisation ``D``
  (passed as a dense ``(n, n)`` diagonal ``b`` with ``l = n``), ``R`` satisfies
  ``Rᴴ R = (original R)ᴴ R + Dᴴ D``, and ``V``/``T`` represent the orthogonal
  ``Q`` for applying ``Qᴴ`` to a right-hand side.

  Args:
    a: ``(..., n, n)`` upper triangular matrix. The strictly lower triangle is
      ignored.
    b: ``(..., m, n)`` pentagonal matrix — an ``(m - l)``-by-``n`` rectangular
      block on top of an ``l``-by-``n`` upper trapezoidal block.
    l: Number of rows of the upper trapezoidal part of ``b``
      (``0 <= l <= min(m, n)``). Defaults to ``min(m, n)``.
    nb: Block size (``1 <= nb <= n``). Defaults to ``min(n, 32)``. Sets the
      panel width and hence the column-block layout of ``T``.

  Returns:
    A tuple ``(R, V, T)``:

    - ``R``: ``(..., n, n)`` — upper triangle holds the triangular factor.
    - ``V``: ``(..., m, n)`` — the Householder reflectors (overwriting ``b``).
    - ``T``: ``(..., nb, n)`` — the block reflector factors, one ``nb``-wide
      triangular block per panel.

  Note:
    On CPU this dispatches to LAPACK ``tpqrt`` (``R, V, T`` exactly as SciPy
    returns them). On other platforms (or under ``vmap``/``grad`` tracing) it
    falls back to a pure-JAX *blocked* Householder factorization (compact-WY,
    the same blocking LAPACK uses). The reflector sign/phase convention of the
    fallback may differ from LAPACK's, so ``R``, ``V`` and ``T`` can differ from
    the CPU path by a per-column unit-modulus phase; each ``(R, V, T)`` triple
    is internally consistent (it reproduces ``C``) and ``Rᴴ R`` is identical.
  """
  a_arr, b_arr = jnp.asarray(a), jnp.asarray(b)
  m, n = b_arr.shape[-2], b_arr.shape[-1]
  if l is None:
    l = min(m, n)
  if nb is None:
    nb = max(1, min(n, 32))
  a, b = core.standard_insert_pvary(a_arr, b_arr)
  return tpqrt_p.bind(a, b, l=int(l), nb=int(nb))


def _tpqrt_shape_rule(a_shape, b_shape, *, l, nb):
  if len(a_shape) != 2 or a_shape[0] != a_shape[1]:
    raise ValueError(
        f"tpqrt requires a square upper-triangular matrix a, got {a_shape}.")
  n = a_shape[1]
  m = b_shape[0]
  if b_shape[1] != n:
    raise ValueError(
        "tpqrt requires b to have the same number of columns as a; "
        f"got a shape {a_shape} and b shape {b_shape}.")
  if not 0 <= l <= min(m, n):
    raise ValueError(
        f"tpqrt requires 0 <= l <= min(m, n); got l={l}, m={m}, n={n}.")
  if n > 0 and not 1 <= nb <= n:
    raise ValueError(f"tpqrt requires 1 <= nb <= n; got nb={nb}, n={n}.")
  return a_shape, b_shape, (nb, n)   # R, V, T


def _tpqrt_pentagonal_mask(m, n, l, dtype):
  """Boolean mask (as ``dtype``) of the pentagonal structure LAPACK tpqrt
  assumes: the bottom ``l`` rows are upper trapezoidal
  (``b[m - l + i, j] = 0`` for ``j < i``)."""
  rows = jnp.arange(m)[:, None]
  cols = jnp.arange(n)[None, :]
  return ((rows < (m - l)) | (cols >= rows - (m - l))).astype(dtype)




def _tpqrt_reflector(alpha, tail, conj, real):
  """Householder reflector ``H = I - tau v vᴴ`` with ``v = [1; v_tail]`` such
  that ``H [alpha; tail] = [beta; 0]``. Returns ``(tau, v_tail)``.

  The reflected pivot ``beta = -(alpha/|alpha|) * ||[alpha; tail]||`` carries the
  phase of ``alpha`` (its sign, for real types); this is what makes the
  reflector annihilate the tail for complex inputs, not just real ones.
  """
  dtype = alpha.dtype
  xnorm2 = real(jnp.sum(conj(tail) * tail))
  absalpha2 = real(conj(alpha) * alpha)
  mu = jnp.sqrt(absalpha2 + xnorm2)
  absalpha = jnp.sqrt(absalpha2)
  phase = jnp.where(absalpha == 0, jnp.ones_like(alpha),
                    alpha / absalpha.astype(dtype))
  beta = -phase * mu.astype(dtype)
  reflect = xnorm2 > 0                  # tail already zero => nothing to do
  safe_beta = jnp.where(reflect, beta, jnp.ones_like(beta))
  safe_den = jnp.where(reflect, alpha - beta, jnp.ones_like(beta))
  tau = jnp.where(reflect, (safe_beta - alpha) / safe_beta,
                  jnp.zeros_like(alpha))
  v_tail = jnp.where(reflect, tail / safe_den, jnp.zeros_like(tail))
  return tau, v_tail


def _tpqrt_householder_2d(a, b, l):
  """Pure-JAX Householder triangularisation of ``[a; b]`` (unbatched).

  The unblocked LAPACK-like fallback. Column ``j`` of the stack ``[a; b]`` has
  all of its sub-diagonal mass in ``b`` (``a`` is already upper triangular), so
  a single Householder reflector mixing the pivot row ``R[j, :]`` with the ``m``
  rows of ``b`` zeros ``b[:, j]``. We ``scan`` over the columns, carrying ``B``
  and *emitting* each finished row of ``R`` — XLA stacks the outputs, so no
  explicit row scatter is needed.
  """
  n = a.shape[-1]
  m = b.shape[-2]
  dtype = a.dtype
  is_complex = dtypes.issubdtype(dtype, np.complexfloating)
  conj = lax.conj if is_complex else (lambda x: x)
  real = lax.real if is_complex else (lambda x: x)

  R = jnp.triu(a)
  B = b * _tpqrt_pentagonal_mask(m, n, l, dtype)

  def body(B, xs):
    j, r_row = xs                          # r_row = original R[j, :]
    tau, v_tail = _tpqrt_reflector(r_row[j], B[:, j], conj, real)
    # vᴴ P over the stacked pivot row and B (columns < j are already zero).
    vHP = r_row + conj(v_tail) @ B
    B = B - tau * v_tail[:, None] * vHP[None, :]
    return B, r_row - tau * vHP

  _, R = jax.lax.scan(body, B, (jnp.arange(n), R))
  return jnp.triu(R)


def _tpqrt_blocked_householder_2d(a, b, l, nb):
  """Pure-JAX *blocked* Householder triangularisation of ``[a; b]`` (unbatched).

  The compact-WY ("``T`` matrix") blocking that LAPACK ``tpqrt``/``tprfb`` use.
  Each reflector's ``R``-component is a unit vector (it pivots on its own row),
  so the panel reflectors stack as ``V = [I; V_B]`` and the block reflector is
  ``Q = I - V T Vᴴ`` with ``T`` upper triangular ``pb x pb``. Per panel:

    1. factor the ``pb``-column panel with unblocked reflectors (BLAS-2, small),
       building ``V_B`` (m x pb) and ``T``;
    2. apply ``Qᴴ = I - V Tᴴ Vᴴ`` to the trailing columns as three matmuls
       (BLAS-3): ``W = R_panelᵀ + V_Bᴴ B``, ``Z = Tᴴ W``,
       ``R_panel -= Z``, ``B -= V_B Z``.

  The dense trailing update is what closes the gap to ``geqrf`` at large ``n``.
  The panel loop is a Python loop (the panel count is static), so all offsets
  and slice widths are statically known — no padding, masking, or dynamic
  slicing. Only the within-panel work is a ``scan`` (over small fixed-size
  blocks), keeping the traced graph ``O(n / nb)`` rather than ``O(n)``.
  Returns ``(R, V, T)`` matching LAPACK tpqrt's outputs: ``R`` (n x n, upper
  triangle), ``V`` (m x n reflectors) and ``T`` (nb x n, one ``pb``-wide
  triangular block per panel at columns ``[p:p+pb]``).
  """
  n = a.shape[-1]
  m = b.shape[-2]
  dtype = a.dtype
  is_complex = dtypes.issubdtype(dtype, np.complexfloating)
  conj = lax.conj if is_complex else (lambda x: x)
  real = lax.real if is_complex else (lambda x: x)

  R = jnp.triu(a)
  B = b * _tpqrt_pentagonal_mask(m, n, l, dtype)
  V = jnp.zeros((m, n), dtype)                  # Householder reflectors
  Tmat = jnp.zeros((nb, n), dtype)              # block reflector factors

  for p in range(0, n, nb):
    pb = min(nb, n - p)                       # static panel width
    Rpp = R[p:p + pb, p:p + pb]               # panel diagonal block (pb x pb)
    Bp = B[:, p:p + pb]                        # panel columns of B (m x pb)
    iota = jnp.arange(pb)

    # 1. Unblocked factorisation of the panel: scan over panel columns,
    #    emitting each finished R row, the reflector tail and tau.
    def col_body(Bp, xs):
      jl, rpp_row = xs
      tau, u = _tpqrt_reflector(rpp_row[jl], Bp[:, jl], conj, real)
      vHP = rpp_row + conj(u) @ Bp              # within-panel only (pb,)
      Bp = Bp - tau * u[:, None] * vHP[None, :]
      return Bp, (rpp_row - tau * vHP, u, tau)

    Bp, (Rpp, Vb_rows, Tau) = jax.lax.scan(col_body, Bp, (iota, Rpp))
    Vb = Vb_rows.T                              # (m, pb) reflector tails

    # 2. Triangular factor T of the block reflector (T[i, j], i <= j).
    def t_body(T, xs):
      j, vj = xs
      x = (conj(Vb).T @ vj) * (iota < j)        # V_Bᴴ v_j, masked to i < j
      col = -Tau[j] * (T @ x)                    # rows >= j vanish (T upper-▲)
      col = jnp.where(iota == j, Tau[j],
                      jnp.where(iota < j, col, jnp.zeros_like(col)))
      return T.at[:, j].set(col), None

    T, _ = jax.lax.scan(t_body, jnp.zeros((pb, pb), dtype), (iota, Vb.T))
    R = R.at[p:p + pb, p:p + pb].set(Rpp)
    V = V.at[:, p:p + pb].set(Vb)
    Tmat = Tmat.at[:pb, p:p + pb].set(T)

    # 3. BLAS-3 trailing update: Qᴴ = I - V Tᴴ Vᴴ on columns [p + pb:].
    if p + pb < n:
      Rtr = R[p:p + pb, p + pb:]               # pivot rows, trailing columns
      Btr = B[:, p + pb:]
      W = Rtr + conj(Vb).T @ Btr                # (pb, n - p - pb)
      Z = conj(T).T @ W
      R = R.at[p:p + pb, p + pb:].set(Rtr - Z)
      B = B.at[:, p + pb:].set(Btr - Vb @ Z)

  return jnp.triu(R), V, Tmat


def _tpqrt_lowering(a, b, *, l, nb):
  f = partial(_tpqrt_blocked_householder_2d, l=l, nb=nb)
  for _ in range(a.ndim - 2):
    f = jax.vmap(f)
  return f(a, b)


def _tpqrt_cpu_lowering(ctx, a, b, *, l, nb):
  a_aval = ctx.avals_in[0]
  target_name = lapack.prepare_lapack_call("tpqrt_ffi", a_aval.dtype)
  # a -> R (output 0), b -> V (output 1); T (output 2) is freshly allocated.
  rule = _linalg_ffi_lowering(target_name,
                              operand_output_aliases={0: 0, 1: 1})
  return rule(ctx, a, b, l=l, nb=nb)


def _tpqrt_dtype_rule(a_dtype, b_dtype, **_):
  return a_dtype, a_dtype, a_dtype   # R, V, T all share the input dtype


tpqrt_p = linalg_primitive(
    _tpqrt_dtype_rule, (_float | _complex, _float | _complex), (2, 2),
    _tpqrt_shape_rule, "tpqrt", multiple_results=True)

# Pure-JAX blocked-Householder fallback (all platforms without a registered FFI
# target, plus vmap/grad tracing).
mlir.register_lowering(tpqrt_p, mlir.lower_fun(
    _tpqrt_lowering, multiple_results=True))

# CPU lowering via LAPACK tpqrt.
mlir.register_lowering(tpqrt_p, _tpqrt_cpu_lowering, platform='cpu')
