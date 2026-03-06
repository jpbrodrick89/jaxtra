"""
jaxtra._src.lax.linalg — ormqr, ldl, and ldl_solve JAX primitives.

Registers primitives using JAX's XLA FFI, backed by LAPACK on CPU and
cuSOLVER on GPU (where available).  The LAPACK FFI targets are registered from
jaxtra's C extension (_jaxtra.so); the GPU targets from _jaxtra_cuda.so when
present.
"""

from __future__ import annotations

import functools

import numpy as np

from jax._src import core, dtypes, dispatch
from jax._src.lax import lax
from jax._src.lax import control_flow
from jax._src.interpreters import mlir, batching
from jax._src.lax.linalg import (
  standard_linalg_primitive,
  linalg_primitive,
  register_cpu_gpu_lowering,
  register_module_custom_calls,
  _linalg_ffi_lowering,
  _float,
  _complex,
  _tril,
  triangular_solve,
  lu as _jax_lu,
)
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import ArrayLike, Array

from jaxtra._src.lib import lapack, gpu_solver


# ---------------------------------------------------------------------------
# Register FFI targets (mirrors jax._src.lax.linalg's pattern)
# ---------------------------------------------------------------------------

register_module_custom_calls(lapack)
register_module_custom_calls(gpu_solver)


# ---------------------------------------------------------------------------
# Orthogonal QR multiply  (verbatim from PR #35104)
# ---------------------------------------------------------------------------


def ormqr(
  a: ArrayLike,
  taus: ArrayLike,
  c: ArrayLike,
  *,
  left: bool = True,
  transpose: bool = False,
) -> Array:
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
      f"the Householder matrix a. Got a shape {a_shape} and c shape {c_shape}."
    )
  if not left and c_shape[1] != m:
    raise ValueError(
      "ormqr with left=False expects c to have the same number of columns as "
      f"the Householder matrix a has rows. Got a shape {a_shape} and c shape {c_shape}."
    )
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
  use_reverse = left != transpose

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
      vHc = lax.dot_general(
        v_h,
        c,
        (((v_h.ndim - 1,), (c.ndim - 2,)), (batch_contract, batch_contract)),
      )
      update = lax.expand_dims(v, (-1,)) * lax.expand_dims(vHc, (-2,))
    else:
      # c = c - tau * (c @ v) @ v^H
      cv = lax.dot_general(
        c, v, (((c.ndim - 1,), (v.ndim - 1,)), (batch_contract, batch_contract))
      )
      v_h = lax.conj(v) if is_complex else v
      update = lax.expand_dims(cv, (-1,)) * lax.expand_dims(v_h, (-2,))
    return c - tau_bc * update

  return control_flow.fori_loop(0, k, body, c)


def _ormqr_cpu_gpu_lowering(
  ctx, a, taus, c, *, left, transpose, target_name_prefix: str
):
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
  (_float | _complex, _float | _complex, _float | _complex),
  (2, 1, 2),
  _ormqr_shape_rule,
  "ormqr",
)
mlir.register_lowering(
  ormqr_p, mlir.lower_fun(_ormqr_lowering, multiple_results=False)
)
register_cpu_gpu_lowering(ormqr_p, _ormqr_cpu_gpu_lowering)


# ---------------------------------------------------------------------------
# LDL decomposition  (symmetric / Hermitian indefinite)
# ---------------------------------------------------------------------------
# Wraps LAPACK ?sytrf (symmetric) / ?hetrf (Hermitian) on CPU and cuSOLVER
# Ssytrf/Dsytrf/Csytrf/Zsytrf on GPU.  Returns (factors, ipiv) where:
#   factors — packed LDL factorization, same dtype and shape as input a.
#   ipiv    — int32 pivot array, shape [..., n].  Uses LAPACK's 1-indexed
#             Bunch-Kaufman format.
# ---------------------------------------------------------------------------


def ldl(
  a: ArrayLike, *, lower: bool = True, hermitian: bool = False
) -> tuple[Array, Array]:
  """LDL factorization (LAPACK sytrf / hetrf).

  Computes the Bunch-Kaufman factorization of a symmetric (``hermitian=False``)
  or Hermitian (``hermitian=True``) indefinite matrix.

  Args:
    a: Square matrix of shape ``[..., n, n]``.
    lower: If ``True`` (default), factorize as ``P @ A @ P^T = L @ D @ L^T``.
      If ``False``, factorize as ``P @ A @ P^T = U^T @ D @ U``.
    hermitian: If ``True``, treat ``a`` as Hermitian (conjugate-symmetric) and
      call LAPACK hetrf (CPU) / sytrf (GPU, which coincides with sytrf for
      Hermitian inputs at the LAPACK level).  For real matrices this flag has
      no effect.

  Returns:
    A tuple ``(factors, ipiv)`` where:

    * ``factors`` — packed factorization stored in the same shape and dtype as
      ``a``.  The strict lower (or upper when ``lower=False``) triangle holds
      the unit-diagonal factor L (or U), and the diagonal (plus one
      sub-/super-diagonal for 2×2 blocks) encodes D.
    * ``ipiv`` — int32 array of shape ``[..., n]`` with LAPACK's 1-indexed
      Bunch-Kaufman pivot information.

  See Also:
    :func:`jaxtra.scipy.linalg.ldl` for a scipy-compatible interface that
    returns ``(lu, d, perm)`` directly.
    :func:`ldl_solve` to solve ``A @ x = b`` given the factorization.
  """
  (a,) = core.standard_insert_pvary(a)
  return ldl_p.bind(a, lower=lower, hermitian=hermitian)


def ldl_solve(
  factors: ArrayLike,
  ipiv: ArrayLike,
  b: ArrayLike,
  *,
  lower: bool = True,
  hermitian: bool = False,
) -> Array:
  """Solve ``A @ x = b`` using an LDL factorization from :func:`ldl`.

  Uses LAPACK ``?sytrs`` (symmetric) or ``?hetrs`` (Hermitian) on CPU via
  jaxtra's XLA FFI binding.  JIT- and vmap-compatible.

  Args:
    factors: Packed LDL factorization from :func:`ldl`, shape ``[..., n, n]``.
    ipiv: Pivot indices from LAPACK sytrf/hetrf, shape ``[..., n]`` (int32).
    b: Right-hand side, shape ``[..., n]`` or ``[..., n, nrhs]``.
    lower: Must match the ``lower`` flag passed to :func:`ldl`.
    hermitian: Must match the ``hermitian`` flag passed to :func:`ldl`.

  Returns:
    Solution ``x`` with the same shape as ``b``.
  """
  import jax.numpy as jnp

  factors = jnp.asarray(factors)
  ipiv = jnp.asarray(ipiv)
  b = jnp.asarray(b)

  (factors, b) = promote_dtypes_inexact(factors, b)

  is_1d = b.ndim == factors.ndim - 1
  if is_1d:
    b = b[..., None]

  factors, ipiv, b = core.standard_insert_pvary(factors, ipiv, b)
  x = ldl_solve_p.bind(factors, ipiv, b, lower=lower, hermitian=hermitian)

  if is_1d:
    x = x[..., 0]
  return x


def _reconstruct_ldl_numpy(factors, ipiv, lower, hermitian):
  """Convert LAPACK sytrf/hetrf output to (L_or_U, D, perm).

  Mirrors scipy.linalg._decomp_ldl logic:
    1. Sanitize ipiv into swap_ (row-swap sequence) and pivots (block sizes).
    2. Extract the block-diagonal D and the raw unit triangular L from the
       packed LAPACK output.
    3. Apply the REVERSE swap sequence to L's rows (as scipy does in
       _ldl_construct_tri_factor) to get an L that satisfies
       L @ D @ L^{H/T} = A (original, non-triangular in general).
    4. Build perm simultaneously so that L[perm, :] is triangular.
    5. Return (L[perm, :], D, perm) so that the caller can check
       lu @ d @ lu^{H/T} == A[perm][:, perm].

  Returns:
    lu:   unit lower (lower=True) or unit upper (lower=False) triangular
          factor with unit diagonal.  Shape ``(n, n)``.
    d:    block-diagonal D matrix (n x n).
    perm: 0-indexed permutation array such that
          ``lu @ d @ lu^{H/T} == A[perm][:, perm]``.
  """
  import numpy as np_

  n = factors.shape[-1]
  is_complex = np_.issubdtype(factors.dtype, np_.complexfloating)

  # ------------------------------------------------------------------
  # Step 1: Sanitize ipiv → (swap_, pivots)
  # Mirrors scipy's _ldl_sanitize_ipiv.
  # swap_[k]  = j means "row k was interchanged with row j".
  # pivots[k] = 1 (1×1 block) or 2 (first entry of 2×2 block) or 0.
  # ------------------------------------------------------------------
  swap_ = np_.arange(n)
  pivots = np_.zeros(n, dtype=int)
  skip_2x2 = False

  # x, y: offsets for checking/writing the 2×2 partner entry.
  # rs/re/ri: iteration range.
  x, y = (1, 0) if lower else (-1, -1)
  rs, re, ri = (0, n, 1) if lower else (n - 1, -1, -1)

  for ind in range(rs, re, ri):
    if skip_2x2:
      skip_2x2 = False
      continue
    cur_val = int(ipiv[ind])
    if cur_val > 0:  # 1×1 block
      if cur_val != ind + 1:  # swap needed (1-indexed check)
        swap_[ind] = swap_[cur_val - 1]
      pivots[ind] = 1
    elif cur_val < 0 and cur_val == int(ipiv[ind + x]):  # 2×2 block
      if -cur_val != ind + 2:  # swap needed for the off-diagonal partner
        swap_[ind + x] = swap_[-cur_val - 1]
      pivots[ind + y] = 2
      skip_2x2 = True
    else:
      raise ValueError("Invalid LAPACK pivot array")

  # ------------------------------------------------------------------
  # Step 2: Extract D and raw unit triangular L
  # Mirrors scipy's _ldl_get_d_and_l.
  # ------------------------------------------------------------------
  d = np_.diag(np_.diag(factors)).astype(factors.dtype)
  lu = (np_.tril(factors, -1) if lower else np_.triu(factors, 1)).copy()
  lu = lu + np_.eye(n, dtype=factors.dtype)  # unit diagonal

  px, py = (1, 0) if lower else (0, 1)
  blk_i = 0
  for blk in pivots[pivots != 0]:
    if blk == 2:
      off = factors[blk_i + px, blk_i + py]
      d[blk_i + px, blk_i + py] = off
      d[blk_i + py, blk_i + px] = (
        np_.conj(off) if (is_complex and hermitian) else off
      )
      lu[blk_i + px, blk_i + py] = 0  # clear 2×2 off-diagonal from L
    blk_i += blk

  # ------------------------------------------------------------------
  # Step 3: Apply reverse swaps to L's rows, tracking perm_arr.
  # Mirrors scipy's _ldl_construct_tri_factor.
  # After this, lu satisfies  lu @ d @ lu^{H/T} = A  (non-triangular).
  # perm = argsort(perm_arr) makes lu[perm, :] triangular.
  # ------------------------------------------------------------------
  perm_arr = np_.arange(n)
  rs2, re2, ri2 = (n - 1, -1, -1) if lower else (0, n, 1)

  for ind in range(rs2, re2, ri2):
    s_ind = swap_[ind]
    if s_ind != ind:
      col_s = ind if lower else 0
      col_e = n if lower else ind + 1
      # For a 2×2 block's second entry (pivots==0 for lower, ==2 for upper):
      # include one extra adjacent column in the row swap.
      if pivots[ind] == (0 if lower else 2):
        col_s += -1 if lower else 0
        col_e += 0 if lower else 1
      lu[[s_ind, ind], col_s:col_e] = lu[[ind, s_ind], col_s:col_e]
      perm_arr[[s_ind, ind]] = perm_arr[[ind, s_ind]]

  # argsort(perm_arr) gives the row permutation that makes lu triangular.
  perm = np_.argsort(perm_arr).astype(np_.int32)

  # ------------------------------------------------------------------
  # Step 4: Reorder rows → triangular lu satisfying
  #   lu @ d @ lu^{H/T} == A[perm][:, perm]
  # ------------------------------------------------------------------
  lu = lu[perm, :]

  # Cast back to real if dtype is real.
  if not is_complex:
    lu = lu.real.astype(factors.dtype)
    d = d.real.astype(factors.dtype)

  return lu, d, perm


# --- Shape / dtype rules ---


def _ldl_shape_rule(shape, *, lower, hermitian):
  if len(shape) < 2 or shape[0] != shape[1]:
    raise ValueError(f"ldl requires a square matrix, got shape {shape}")
  n = shape[0]
  return shape, (n,)


def _ldl_dtype_rule(dtype, **_):
  return dtype, dtypes.dtype(np.int32)


# --- Python fallback (LU-based) ---


def _ldl_python_impl(a, *, lower, hermitian):
  """Pure-JAX fallback: use LU decomposition on platforms without sytrf."""
  lu_result, pivots, _ = _jax_lu(a)
  return lu_result, pivots


# --- CPU/GPU FFI lowering ---


def _ldl_cpu_gpu_lowering(ctx, a, *, lower, hermitian, target_name_prefix: str):
  (a_aval,) = ctx.avals_in
  factors_aval, ipiv_aval = ctx.avals_out
  dtype = a_aval.dtype

  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("sytrf_ffi", dtype)
  else:
    # GPU: cuSolver sytrf for all dtypes (symmetric).
    # For complex Hermitian on GPU there is no cuSolver hetrf; fall back to LU.
    if hermitian and dtypes.issubdtype(dtype, np.complexfloating):
      return mlir.lower_fun(_ldl_python_impl, multiple_results=True)(
        ctx, a, lower=lower, hermitian=hermitian
      )
    target_name = f"{target_name_prefix}solver_sytrf_ffi"

  rule = _linalg_ffi_lowering(
    target_name,
    avals_out=[factors_aval, ipiv_aval],
    operand_output_aliases={0: 0},
  )
  return rule(ctx, a, lower=lower, hermitian=hermitian)


# --- Primitive registration ---

ldl_p = linalg_primitive(
  _ldl_dtype_rule,
  (_float | _complex,),
  (2,),
  _ldl_shape_rule,
  "ldl",
  multiple_results=True,
)
mlir.register_lowering(
  ldl_p, mlir.lower_fun(_ldl_python_impl, multiple_results=True)
)
register_cpu_gpu_lowering(ldl_p, _ldl_cpu_gpu_lowering)


# ---------------------------------------------------------------------------
# LDL solve  (sytrs / hetrs)
# ---------------------------------------------------------------------------
# JIT- and vmap-compatible solve using LAPACK ?sytrs / ?hetrs on CPU.
# Takes (factors, ipiv, b) from ldl() and returns x such that A @ x ≈ b.
# GPU falls back to a JAX triangular-solve approximation (no cuSOLVER sytrs).
# ---------------------------------------------------------------------------


def _ldl_solve_abstract_eval(
  factors_aval, ipiv_aval, b_aval, *, lower, hermitian
):
  return core.ShapedArray(b_aval.shape, factors_aval.dtype)


def _ldl_solve_jax_fallback(factors, ipiv, b, *, lower, hermitian):
  """JAX-traceable fallback: approximate solve via unit-triangular factors.

  This path is used on platforms without the sytrs FFI target (e.g. GPU,
  TPU).  It ignores the D factor and the permutation, so it is only
  approximate, but gives the correct dtype/shape.
  """
  import jax.numpy as jnp

  n = factors.shape[-1]
  eye = lax._eye(factors.dtype, (n, n))
  if factors.ndim > 2:
    eye = lax.broadcast(eye, factors.shape[:-2])
  if lower:
    L = _tril(factors, -1) + eye
    y = triangular_solve(L, b, left_side=True, lower=True, unit_diagonal=True)
    L_H = jnp.conj(L).swapaxes(-1, -2) if hermitian else L.swapaxes(-1, -2)
    return triangular_solve(
      L_H, y, left_side=True, lower=False, unit_diagonal=True
    )
  else:
    U = jnp.triu(factors, 1) + eye
    y = triangular_solve(U, b, left_side=True, lower=False, unit_diagonal=True)
    U_H = jnp.conj(U).swapaxes(-1, -2) if hermitian else U.swapaxes(-1, -2)
    return triangular_solve(
      U_H, y, left_side=True, lower=True, unit_diagonal=True
    )


def _ldl_solve_cpu_gpu_lowering(
  ctx, factors, ipiv, b, *, lower, hermitian, target_name_prefix: str
):
  factors_aval, _, _ = ctx.avals_in
  (x_aval,) = ctx.avals_out
  dtype = factors_aval.dtype

  if target_name_prefix == "cpu":
    target_name = lapack.prepare_lapack_call("sytrs_ffi", dtype)
    rule = _linalg_ffi_lowering(
      target_name,
      avals_out=[x_aval],
      operand_output_aliases={2: 0},
    )
    return rule(ctx, factors, ipiv, b, lower=lower, hermitian=hermitian)
  else:
    # No cuSOLVER sytrs equivalent; fall back to triangular solve.
    return mlir.lower_fun(_ldl_solve_jax_fallback, multiple_results=False)(
      ctx, factors, ipiv, b, lower=lower, hermitian=hermitian
    )


def _ldl_solve_batching_rule(batched_args, batch_dims, *, lower, hermitian):
  import jax.numpy as jnp

  factors, ipiv, b = batched_args
  f_bd, i_bd, b_bd = batch_dims

  factors = (
    batching.moveaxis(factors, f_bd, 0)
    if f_bd is not None
    else jnp.expand_dims(factors, 0)
  )
  ipiv = (
    batching.moveaxis(ipiv, i_bd, 0)
    if i_bd is not None
    else jnp.expand_dims(ipiv, 0)
  )
  b = (
    batching.moveaxis(b, b_bd, 0)
    if b_bd is not None
    else jnp.expand_dims(b, 0)
  )

  x = ldl_solve_p.bind(factors, ipiv, b, lower=lower, hermitian=hermitian)
  return x, 0


ldl_solve_p = core.Primitive("ldl_solve")
ldl_solve_p.multiple_results = False
ldl_solve_p.def_abstract_eval(_ldl_solve_abstract_eval)
ldl_solve_p.def_impl(functools.partial(dispatch.apply_primitive, ldl_solve_p))
mlir.register_lowering(
  ldl_solve_p,
  mlir.lower_fun(_ldl_solve_jax_fallback, multiple_results=False),
)
register_cpu_gpu_lowering(ldl_solve_p, _ldl_solve_cpu_gpu_lowering)
batching.primitive_batchers[ldl_solve_p] = _ldl_solve_batching_rule
