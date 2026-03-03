"""
jaxtra._src.lax.linalg — ormqr and pentadiagonal_solve JAX primitives.

Registers primitives using JAX's XLA FFI, backed by LAPACK (CPU) or
cuSOLVER/cuSPARSE (GPU).  The LAPACK FFI targets are registered from
jaxtra's C extension (_jaxtra.so); the GPU targets from _jaxtra_cuda.so
when present.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from jax._src import core, dtypes
try:
  from jax import ffi as _jax_ffi  # JAX >= 0.4.36 public API
except ImportError:
  from jax.extend import ffi as _jax_ffi  # JAX 0.4.27–0.4.35
from jax._src.lax import lax
from jax._src.lax import control_flow
from jax._src.interpreters import mlir, ad
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
    a: The Householder reflectors from :func:`geqrf` or :func:`geqp3`,
      with shape ``[..., m, n]``.
    taus: The Householder scalar factors from :func:`geqrf` or :func:`geqp3`,
      with shape ``[..., k]``.
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


def _penta_matvec(ds, dl, d, du, dw, v):
  """Apply the pentadiagonal matrix A to vector v: returns A @ v."""
  zeros_1 = jnp.zeros_like(v[..., :1])
  zeros_2 = jnp.zeros_like(v[..., :2])
  # Row i contribution: ds[i]*v[i-2] + dl[i]*v[i-1] + d[i]*v[i]
  #                   + du[i]*v[i+1] + dw[i]*v[i+2]
  out = d * v
  out = out + jnp.concatenate([zeros_1, dl[..., 1:] * v[..., :-1]], axis=-1)
  out = out + jnp.concatenate([zeros_2, ds[..., 2:] * v[..., :-2]], axis=-1)
  out = out + jnp.concatenate([du[..., :-1] * v[..., 1:], zeros_1], axis=-1)
  out = out + jnp.concatenate([dw[..., :-2] * v[..., 2:], zeros_2], axis=-1)
  return out


def pentadiagonal_solve(
    ds: ArrayLike, dl: ArrayLike, d: ArrayLike,
    du: ArrayLike, dw: ArrayLike, b: ArrayLike) -> Array:
  """Solve a pentadiagonal linear system A x = b.

  The matrix A is stored as five diagonals following the cuSPARSE
  ``gpsvInterleavedBatch`` convention::

      A[i, i-2] = ds[i]   (ds[0], ds[1] unused)
      A[i, i-1] = dl[i]   (dl[0] unused)
      A[i, i]   = d[i]
      A[i, i+1] = du[i]   (du[n-1] unused)
      A[i, i+2] = dw[i]   (dw[n-2], dw[n-1] unused)

  Args:
    ds: Second sub-diagonal of A with shape ``[..., n]``.
    dl: First sub-diagonal of A with shape ``[..., n]``.
    d: Main diagonal of A with shape ``[..., n]``.
    du: First super-diagonal of A with shape ``[..., n]``.
    dw: Second super-diagonal of A with shape ``[..., n]``.
    b: Right-hand side vector with shape ``[..., n]``.

  Returns:
    Solution x with the same shape as b.
  """
  ds, dl, d, du, dw, b = core.standard_insert_pvary(ds, dl, d, du, dw, b)
  return pentadiagonal_solve_p.bind(ds, dl, d, du, dw, b)


def _pentadiagonal_solve_shape_rule(
    ds_shape, dl_shape, d_shape, du_shape, dw_shape, b_shape):
  n = d_shape[0]
  for name, shape in [("ds", ds_shape), ("dl", dl_shape),
                      ("du", du_shape), ("dw", dw_shape)]:
    if shape[0] != n:
      raise ValueError(
          f"pentadiagonal_solve: diagonal {name} has shape {shape}, "
          f"expected shape matching d shape ({n},)")
  if b_shape[0] != n:
    raise ValueError(
        f"pentadiagonal_solve: b has shape {b_shape}, "
        f"expected leading dim {n} to match d")
  return b_shape


def _pentadiagonal_solve_jax_fallback(ds, dl, d, du, dw, b):
  """Pure-JAX fallback: reconstruct dense matrix and use LU solve."""
  n = d.shape[-1]
  # Build full n x n matrix from the five diagonals.
  A = jnp.diag(d)
  A = A.at[jnp.arange(1, n), jnp.arange(n - 1)].add(dl[1:])
  A = A.at[jnp.arange(2, n), jnp.arange(n - 2)].add(ds[2:])
  A = A.at[jnp.arange(n - 1), jnp.arange(1, n)].add(du[:n - 1])
  A = A.at[jnp.arange(n - 2), jnp.arange(2, n)].add(dw[:n - 2])
  return jnp.linalg.solve(A, b)


def _pentadiagonal_solve_cpu_gpu_lowering(
    ctx, ds, dl, d, du, dw, b, *, target_name_prefix: str):
  d_aval = ctx.avals_in[2]
  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("gbsv_ffi", d_aval.dtype)
  else:
    target_name = f"{target_name_prefix}sparse_gpsvInterleaved_ffi"
  rule = _jax_ffi.ffi_lowering(target_name, operand_output_aliases={0: 5})
  return rule(ctx, ds, dl, d, du, dw, b)


def _pentadiagonal_solve_jvp(primals, tangents):
  ds, dl, d, du, dw, b = primals
  dds, ddl, dd, ddu, ddw, db = tangents

  x = pentadiagonal_solve_p.bind(ds, dl, d, du, dw, b)

  # Replace Zero tangents with actual zeros.
  z = lambda t, ref: jnp.zeros_like(ref) if type(t) is ad.Zero else t
  dds = z(dds, ds); ddl = z(ddl, dl); dd = z(dd, d)
  ddu = z(ddu, du); ddw = z(ddw, dw); db = z(db, b)

  # Compute dA @ x (matvec of the perturbation matrix applied to x).
  dAx = _penta_matvec(dds, ddl, dd, ddu, ddw, x)

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
  zeros_1 = jnp.zeros(d.shape[:-1] + (1,), dtype=d.dtype)
  zeros_2 = jnp.zeros(d.shape[:-1] + (2,), dtype=d.dtype)

  # Build A^T diagonals.
  ds_T = jnp.concatenate([zeros_2, dw[..., :-2]], axis=-1)   # lower-2
  dl_T = jnp.concatenate([zeros_1, du[..., :-1]], axis=-1)   # lower-1
  d_T = d                                                      # main
  du_T = jnp.concatenate([dl[..., 1:], zeros_1], axis=-1)    # upper-1
  dw_T = jnp.concatenate([ds[..., 2:], zeros_2], axis=-1)    # upper-2

  ct_b = pentadiagonal_solve_p.bind(ds_T, dl_T, d_T, du_T, dw_T, ct)
  return [None, None, None, None, None, ct_b]


pentadiagonal_solve_p = standard_linalg_primitive(
    (_float | _complex,) * 6, (1, 1, 1, 1, 1, 1),
    _pentadiagonal_solve_shape_rule, "pentadiagonal_solve")

# Register pure-JAX fallback lowering (used on TPU / platforms without FFI).
mlir.register_lowering(pentadiagonal_solve_p, mlir.lower_fun(
    _pentadiagonal_solve_jax_fallback, multiple_results=False))

# Register CPU (LAPACK gbsv) and GPU (cuSPARSE gpsvInterleavedBatch) lowerings.
register_cpu_gpu_lowering(pentadiagonal_solve_p,
                          _pentadiagonal_solve_cpu_gpu_lowering)

# Register JVP and transpose rules.
ad.primitive_jvps[pentadiagonal_solve_p] = _pentadiagonal_solve_jvp
ad.primitive_transposes[pentadiagonal_solve_p] = _pentadiagonal_solve_transpose
