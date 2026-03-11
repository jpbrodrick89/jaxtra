"""
jaxtra._src.lax.linalg — ormqr, ldl, ldl_solve, pentadiagonal_solve,
and pentadiagonal_solveh JAX primitives.

Registers primitives using JAX's XLA FFI, backed by LAPACK on CPU and
cuSOLVER on GPU (where available).  The LAPACK FFI targets are registered from
jaxtra's C extension (_jaxtra.so); the GPU targets from _jaxtra_cuda.so when
present.
"""

from __future__ import annotations

import functools
from functools import partial

import numpy as np

import jax.numpy as jnp
from jax import ffi as _jax_ffi
from jax._src import core, dtypes, dispatch
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
    triangular_solve,
    lu as _jax_lu,
)
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import ArrayLike, Array

from jaxtra._src.lib import lapack, gpu_solver, gpu_sparse


# ---------------------------------------------------------------------------
# Register FFI targets (mirrors jax._src.lax.linalg's pattern)
# ---------------------------------------------------------------------------

register_module_custom_calls(lapack)
register_module_custom_calls(gpu_solver)
register_module_custom_calls(gpu_sparse)


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
