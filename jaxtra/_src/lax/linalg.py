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

import jax
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

from jaxtra._src.lib import lapack, gpu_solver, gpu_sparse, gpu_hybrid


# ---------------------------------------------------------------------------
# Register FFI targets (mirrors jax._src.lax.linalg's pattern)
# ---------------------------------------------------------------------------

register_module_custom_calls(lapack)
register_module_custom_calls(gpu_solver)
register_module_custom_calls(gpu_sparse)
register_module_custom_calls(gpu_hybrid)


# ---------------------------------------------------------------------------
# LDL decomposition  (symmetric / Hermitian indefinite)
# ---------------------------------------------------------------------------
# CPU: LAPACK ?sytrf (symmetric) or ?hetrf (Hermitian).
# GPU: cuSOLVER sytrf (symmetric); LAPACK ?hetrf via CPU hybrid (Hermitian,
#      cuSOLVER has no hetrf — data is copied D→H, hetrf called, H→D).
# Returns (factors_untangled, ipiv, perm):
#   factors_untangled — BK-corrected L in strict triangle, D on diag/±1.
#   ipiv    — int32 pivot array, shape [..., n].  LAPACK 1-indexed BK format.
#   perm    — int32 permutation array, shape [..., n].
# ---------------------------------------------------------------------------


def ldl(
  a: ArrayLike, *, lower: bool = True, hermitian: bool = False
) -> tuple[Array, Array, Array]:
  """LDL factorization of a symmetric or Hermitian indefinite matrix.

  Computes the Bunch-Kaufman factorization of a symmetric (``hermitian=False``)
  or Hermitian (``hermitian=True``) indefinite matrix using LAPACK ``?sytrf``
  (symmetric) or ``?hetrf`` (Hermitian) on CPU, and cuSOLVER ``sytrf`` on GPU.

  This function returns a packed ``(factors, ipiv, perm)`` result suitable for
  direct use with :func:`jaxtra.lax.linalg.ldl_solve`.  For a decomposition
  that returns the ``(lu, d, perm)`` matrices ready for inspection or
  reconstruction, prefer :func:`jaxtra.scipy.linalg.ldl`.

  Args:
    a: Square matrix of shape ``[..., n, n]``.
    lower: If ``True`` (default), factorize as ``P @ A @ P^T = L @ D @ L^T``.
      If ``False``, factorize as ``P @ A @ P^T = U^T @ D @ U``.
    hermitian: If ``True``, treat ``a`` as Hermitian (conjugate-symmetric).
      For complex dtypes, calls LAPACK ``?hetrf`` on both CPU and GPU (cuSOLVER
      has no hetrf; the GPU path copies data host→device, calls LAPACK, then
      copies back).  For symmetric real/complex-symmetric matrices
      (``hermitian=False``), uses LAPACK ``?sytrf`` on CPU and cuSOLVER
      ``sytrf`` on GPU.  For real matrices this flag has no effect.

  Returns:
    A tuple ``(factors, ipiv, perm)`` where:

    * ``factors`` — packed Bunch-Kaufman factorization, same shape and dtype
      as ``a``.  The strict lower (or upper when ``lower=False``) triangle
      holds the unit-diagonal L (or U) multipliers with Bunch-Kaufman row
      corrections applied; the diagonal plus the ±1 off-diagonal encodes D.
    * ``ipiv`` — int32 array of shape ``[..., n]`` with LAPACK's 1-indexed
      Bunch-Kaufman pivot information.
    * ``perm`` — int32 permutation array of shape ``[..., n]`` derived from
      ``ipiv`` (precomputed at factorization time for efficient use in
      :func:`jaxtra.lax.linalg.ldl_solve`).

  See Also:
    :func:`jaxtra.lax.linalg.ldl_solve`: Solve ``A @ x = b`` given ``(factors, ipiv, perm)``.
    :func:`jaxtra.scipy.linalg.ldl`: SciPy-compatible interface returning ``(lu, d, perm)`` directly.
    :func:`jax.scipy.linalg.lu_factor`: Analogous LU factorization.
    :func:`jax.scipy.linalg.cho_factor`: Analogous Cholesky factorization (positive-definite only).

  Examples:
    Factorize and solve a symmetric indefinite system:

    >>> import jax.numpy as jnp
    >>> from jaxtra.lax.linalg import ldl, ldl_solve
    >>> A = jnp.array([[2., 1.], [1., -3.]])
    >>> factors, ipiv, perm = ldl(A)
    >>> b = jnp.array([1., 2.])
    >>> x = ldl_solve(factors, ipiv, perm, b)
  """
  (a,) = core.standard_insert_pvary(a)
  return ldl_p.bind(a, lower=lower, hermitian=hermitian)


def ldl_solve(
  factors: ArrayLike,
  ipiv: ArrayLike,
  perm: ArrayLike,
  b: ArrayLike,
  *,
  lower: bool = True,
  hermitian: bool = False,
) -> Array:
  """Solve a linear system using an LDL factorization.

  Uses the output of :func:`jaxtra.lax.linalg.ldl` to solve ``A @ x = b``
  where ``A`` is symmetric or Hermitian indefinite.  Implemented as a
  pure-JAX triangular solve (permute → L⁻¹ → D⁻¹ → L⁻ᴴ → unpermute),
  dispatching to cuBLAS triangular solves on GPU.  No LAPACK sytrs/hetrs call
  is made.  JIT-, vmap-, and batching-compatible.

  Args:
    factors: Packed LDL factorization from :func:`jaxtra.lax.linalg.ldl`,
      shape ``[..., n, n]``.
    ipiv: Pivot indices from :func:`jaxtra.lax.linalg.ldl`, shape
      ``[..., n]`` (int32).
    perm: Permutation from :func:`jaxtra.lax.linalg.ldl`, shape
      ``[..., n]`` (int32).
    b: Right-hand side, shape ``[..., n]`` or ``[..., n, nrhs]``.
    lower: Must match the ``lower`` flag passed to :func:`jaxtra.lax.linalg.ldl`.
    hermitian: Must match the ``hermitian`` flag passed to
      :func:`jaxtra.lax.linalg.ldl`.

  Returns:
    Array of shape ``[..., n]`` or ``[..., n, nrhs]`` representing ``x``.

  See Also:
    :func:`jaxtra.lax.linalg.ldl`: Compute the LDL factorization.
    :func:`jaxtra.scipy.linalg.ldl`: SciPy-compatible interface returning ``(lu, d, perm)`` directly.
    :func:`jax.scipy.linalg.lu_solve`: Analogous solve from an LU factorization.
    :func:`jax.scipy.linalg.cho_solve`: Analogous solve from a Cholesky factorization.

  Examples:
    Solve a symmetric indefinite system ``A @ x = b``:

    >>> import jax.numpy as jnp
    >>> from jaxtra.lax.linalg import ldl, ldl_solve
    >>> A = jnp.array([[2., 1., 0.],
    ...                [1., -3., 1.],
    ...                [0., 1., 4.]])
    >>> b = jnp.array([1., 2., 3.])
    >>> factors, ipiv, perm = ldl(A)
    >>> x = ldl_solve(factors, ipiv, perm, b)

    Solve a Hermitian indefinite system (complex128):

    >>> import jax.numpy as jnp
    >>> from jaxtra.lax.linalg import ldl, ldl_solve
    >>> A = jnp.array([[2.+0j, 1.+1j], [1.-1j, -3.+0j]])
    >>> factors, ipiv, perm = ldl(A, hermitian=True)
    >>> x = ldl_solve(factors, ipiv, perm, jnp.array([1.+0j, 2.+0j]),
    ...               hermitian=True)

  .. rubric:: Benchmarks

  Full solve (factorization + triangular solve) of symmetric/Hermitian
  indefinite systems (float64 / complex128, n = 50–5000) compared against
  :func:`jax.scipy.linalg.solve` (dense LU).

  **Real symmetric** — On Linux (OpenBLAS), LDL is **1.5–2.3×** faster at
  small sizes (n ≤ 200), roughly equal at n = 500–1000, and **1.35–1.64×**
  faster at large sizes (n ≥ 2000).  On macOS (Accelerate), ``getrf`` is more
  aggressively optimised than ``sytrf`` so LU may be faster at medium sizes —
  benchmark both paths on your hardware.

  **Complex Hermitian** — On GPU, factorization uses LAPACK ``?hetrf`` via a
  CPU hybrid path (no cuSOLVER hetrf); the triangular solve dispatches to
  cuBLAS and is fast.  At large n the hetrf cost dominates; see benchmarks.

  .. figure:: /_bench_images/bench_ldl.png
     :alt: LDL vs LU full solve benchmark
     :width: 90%
     :align: center

  See ``benchmarks/bench_ldl.py`` for reproduction.
  """
  factors = jnp.asarray(factors)
  ipiv = jnp.asarray(ipiv)
  perm = jnp.asarray(perm)
  b = jnp.asarray(b)

  (factors, b) = promote_dtypes_inexact(factors, b)

  is_1d = b.ndim == factors.ndim - 1
  if is_1d:
    b = b[..., None]

  factors, ipiv, perm, b = core.standard_insert_pvary(factors, ipiv, perm, b)
  x = ldl_solve_p.bind(factors, ipiv, perm, b, lower=lower, hermitian=hermitian)

  if is_1d:
    x = x[..., 0]
  return x


# ---------------------------------------------------------------------------
# ldl_ipiv_to_permutation — convert LAPACK ipiv to permutation array
# ---------------------------------------------------------------------------
# CPU/TPU: pure-JAX fori_loop (sequential swaps, data-dependent carry).
# GPU: single CUDA kernel launch (LdlIpivToPermutationFfi) — one thread per
#      batch element runs the entire swap loop, avoiding O(n) kernel launches.
# ---------------------------------------------------------------------------


def _ldl_ipiv_to_permutation_jax(ipiv, *, lower):
  """Pure-JAX ipiv → permutation (CPU/TPU fallback). Handles batch dims."""
  if ipiv.ndim > 1:
    return jax.vmap(partial(_ldl_ipiv_to_permutation_jax, lower=lower))(ipiv)
  n = ipiv.shape[0]
  perm = jnp.arange(n, dtype=jnp.int32)

  def swap_at(p, i, j):
    pi, pj = p[i], p[j]
    return p.at[i].set(pj).at[j].set(pi)

  if lower:
    def body(k, carry):
      perm, skip = carry
      pk = ipiv[k]
      j_1x1 = pk - 1
      j_2x2 = -pk - 1
      kp1 = jnp.minimum(k + 1, n - 1)
      new_perm = jnp.where(pk > 0,
          swap_at(perm, k, j_1x1),
          jnp.where(~skip, swap_at(perm, kp1, j_2x2), perm))
      new_skip = (pk < 0) & ~skip
      return new_perm, new_skip
    perm, _ = control_flow.fori_loop(0, n, body, (perm, jnp.bool_(False)))
  else:
    def body(k, carry):
      perm, skip = carry
      idx = n - 1 - k
      pk = ipiv[idx]
      j_1x1 = pk - 1
      j_2x2 = -pk - 1
      idxm1 = jnp.maximum(idx - 1, 0)
      new_perm = jnp.where(pk > 0,
          swap_at(perm, idx, j_1x1),
          jnp.where(~skip, swap_at(perm, idxm1, j_2x2), perm))
      new_skip = (pk < 0) & ~skip
      return new_perm, new_skip
    perm, _ = control_flow.fori_loop(0, n, body, (perm, jnp.bool_(False)))

  return perm


def _ldl_ipiv_to_permutation(ipiv, *, lower):
  """Call ldl_ipiv_to_permutation_p.bind — dispatches to CUDA kernel on GPU."""
  return ldl_ipiv_to_permutation_p.bind(ipiv, lower=lower)


def _ldl_ipiv_to_permutation_abstract_eval(ipiv_aval, *, lower):
  return core.ShapedArray(ipiv_aval.shape, jnp.int32)


ldl_ipiv_to_permutation_p = core.Primitive("ldl_ipiv_to_permutation")
ldl_ipiv_to_permutation_p.def_abstract_eval(_ldl_ipiv_to_permutation_abstract_eval)
ldl_ipiv_to_permutation_p.def_impl(
    functools.partial(dispatch.apply_primitive, ldl_ipiv_to_permutation_p))
mlir.register_lowering(
    ldl_ipiv_to_permutation_p,
    mlir.lower_fun(_ldl_ipiv_to_permutation_jax, multiple_results=False))
# GPU: single CUDA kernel launch (one thread per batch element).
mlir.register_lowering(
    ldl_ipiv_to_permutation_p,
    _linalg_ffi_lowering("cu_ldl_ipiv_to_permutation",
                         num_non_batch_dims=1, column_major=False,
                         operand_output_aliases={}),
    platform='cuda')


# --- Shape / dtype rules ---


def _ldl_shape_rule(shape, *, lower, hermitian):
  if len(shape) < 2 or shape[0] != shape[1]:
    raise ValueError(f"ldl requires a square matrix, got shape {shape}")
  n = shape[0]
  return shape, (n,), (n,)


def _ldl_dtype_rule(dtype, **_):
  return dtype, dtypes.dtype(np.int32), dtypes.dtype(np.int32)


# --- Python fallback (LU-based) ---


def _ldl_python_impl(a, *, lower, hermitian):
  """Pure-JAX fallback: use LU decomposition on platforms without sytrf."""
  lu_result, pivots, _ = _jax_lu(a)
  # _jax_lu returns 0-based pivot indices; sytrf uses 1-based.
  ipiv = pivots + 1
  perm = _ldl_ipiv_to_permutation_jax(ipiv, lower=lower)
  factors_untangled = _bk_untangle(lu_result, ipiv, lower)
  return factors_untangled, ipiv, perm


# --- CPU/GPU FFI lowering ---


def _ldl_cpu_gpu_lowering(ctx, a, *, lower, hermitian, target_name_prefix: str):
  (a_aval,) = ctx.avals_in
  factors_aval, ipiv_aval, perm_aval = ctx.avals_out
  dtype = a_aval.dtype

  use_hetrf = hermitian and dtypes.issubdtype(dtype, np.complexfloating)

  if target_name_prefix == "cpu":
    # CPU: separate sytrf_ffi (symmetric) and hetrf_ffi (Hermitian) targets.
    # Real types always use sytrf (shetrf/dhetrf don't exist in LAPACK).
    fn_base = "hetrf_ffi" if use_hetrf else "sytrf_ffi"
    target_name = lapack.prepare_lapack_call(fn_base, dtype)
  else:
    # GPU: cuSolver sytrf for real + complex symmetric.
    # No cuSolver hetrf; use hybrid CPU-LAPACK kernel for complex Hermitian.
    if use_hetrf:
      char = "c" if dtype == np.complex64 else "z"
      target_name = f"{target_name_prefix}hybrid_{char}hetrf_ffi"
    else:
      target_name = f"{target_name_prefix}solver_sytrf_ffi"

  rule = _linalg_ffi_lowering(
    target_name,
    avals_out=[factors_aval, ipiv_aval],
    operand_output_aliases={0: 0},
  )
  # Neither sytrf_ffi/hetrf_ffi (CPU) nor cusolver_sytrf_ffi (GPU) take a
  # hermitian attribute — the choice is encoded in the target name.
  factors, ipiv = rule(ctx, a, lower=lower)

  # Apply BK row-swap corrections so the primitive returns factors_untangled.
  factors, = mlir.lower_fun(
      partial(_bk_untangle, lower=lower),
      multiple_results=False,
  )(ctx.replace(primitive=None, avals_in=[factors_aval, ipiv_aval], avals_out=[factors_aval]), factors, ipiv)

  # Compute permutation from ipiv inline (mirrors _lu_cpu_gpu_lowering calling
  # lu_pivots_to_permutation_p inline to return permutation as a 3rd output).
  # We call through the primitive so the CUDA kernel is used on GPU.
  perm, = mlir.lower_fun(
      partial(_ldl_ipiv_to_permutation, lower=lower),
      multiple_results=False,
  )(ctx.replace(primitive=None, avals_in=[ipiv_aval], avals_out=[perm_aval]), ipiv)

  return factors, ipiv, perm


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
# ldl_raw — raw sytrf/hetrf FFI call (no BK correction, no permutation)
# Used to measure the bare LAPACK/cuSolver cost for benchmarking.
# ---------------------------------------------------------------------------


def _ldl_raw_shape_rule(shape, *, lower, hermitian):
  if len(shape) < 2 or shape[0] != shape[1]:
    raise ValueError(f"ldl_raw requires a square matrix, got shape {shape}")
  n = shape[0]
  return shape, (n,)


def _ldl_raw_dtype_rule(dtype, **_):
  return dtype, dtypes.dtype(np.int32)


def _ldl_raw_python_impl(a, *, lower, hermitian):
  lu_result, pivots, _ = _jax_lu(a)
  return lu_result, pivots + 1


def _ldl_raw_cpu_gpu_lowering(ctx, a, *, lower, hermitian, target_name_prefix: str):
  (a_aval,) = ctx.avals_in
  factors_aval, ipiv_aval = ctx.avals_out
  dtype = a_aval.dtype
  use_hetrf = hermitian and dtypes.issubdtype(dtype, np.complexfloating)
  if target_name_prefix == "cpu":
    fn_base = "hetrf_ffi" if use_hetrf else "sytrf_ffi"
    target_name = lapack.prepare_lapack_call(fn_base, dtype)
  else:
    if use_hetrf:
      char = "c" if dtype == np.complex64 else "z"
      target_name = f"{target_name_prefix}hybrid_{char}hetrf_ffi"
    else:
      target_name = f"{target_name_prefix}solver_sytrf_ffi"
  rule = _linalg_ffi_lowering(
    target_name,
    avals_out=[factors_aval, ipiv_aval],
    operand_output_aliases={0: 0},
  )
  return rule(ctx, a, lower=lower)


ldl_raw_p = linalg_primitive(
  _ldl_raw_dtype_rule,
  (_float | _complex,),
  (2,),
  _ldl_raw_shape_rule,
  "ldl_raw",
  multiple_results=True,
)
mlir.register_lowering(ldl_raw_p, mlir.lower_fun(_ldl_raw_python_impl, multiple_results=True))
register_cpu_gpu_lowering(ldl_raw_p, _ldl_raw_cpu_gpu_lowering)


def ldl_raw(a, *, lower=True, hermitian=None):
  """Raw LAPACK sytrf/hetrf call without BK correction.

  Returns ``(factors_raw, ipiv)`` directly from LAPACK without the
  associative-scan BK correction applied by :func:`ldl`.  Useful for
  benchmarking the LAPACK call cost in isolation.
  """
  a = jnp.asarray(a)
  if hermitian is None:
    hermitian = jnp.issubdtype(a.dtype, jnp.complexfloating)
  (a,) = promote_dtypes_inexact(a)
  return ldl_raw_p.bind(a, lower=lower, hermitian=bool(hermitian))


# ---------------------------------------------------------------------------
# LDL solve  (pure-JAX triangular solve)
# ---------------------------------------------------------------------------
# JIT- and vmap-compatible: P b → L⁻¹ → D⁻¹ → L⁻ᴴ → P^T x.
# Dispatches to cuBLAS triangular solves on GPU.  No LAPACK sytrs/hetrs used.
# ---------------------------------------------------------------------------


def _ldl_solve_abstract_eval(
  factors_aval, ipiv_aval, perm_aval, b_aval, *, lower, hermitian
):
  return core.ShapedArray(b_aval.shape, factors_aval.dtype)


def _bk_perm_compose(left, right):
  """Associative combine for BK permutation composition.

  associative_scan appends an internal span axis as the LAST axis, so inside
  this function left/right have shape (..., n, span).  The row axis is
  second-to-last: ndim - 2.

  Computes the composed gather: right[left[j]] for each row index j, i.e.
  apply left's index-map first, then right's.
  """
  return jnp.take_along_axis(right, left, axis=left.ndim - 2)


def _bk_untangle(factors, ipiv, lower):
  """Apply BK row-swap corrections to the full packed factor matrix.

  LAPACK's BK factorization stores raw L multipliers before subsequent row
  swaps (steps k+1…n-1 for lower, k-1…0 for upper) have been applied to
  column k.  This function returns ``factors_untangled`` where the L
  multipliers in the strict lower/upper triangle are corrected.

  The diagonal (D[k,k]) and 2×2 D off-diagonal entries are guaranteed to be
  unchanged because:
    * LAPACK BK swaps at step j only involve rows >= j (within the active
      submatrix), so P_correct[:,k] leaves rows <= k fixed.
    * The 2×2 D off-diagonal at row k+1 is also fixed: P(k+1) is a no-op
      (second half of the 2×2 pair) and subsequent P(j>=k+2) only touch
      rows >= k+2; bijectivity prevents any L value from landing at row k+1.

  Therefore we can apply P_correct directly to ``factors`` without isolating
  the L multipliers first.
  """
  n = factors.shape[-1]
  is_neg = ipiv < 0  # [..., n]
  pad_end = jnp.zeros((*is_neg.shape[:-1], 1), dtype=bool)
  pad_beg = jnp.zeros((*is_neg.shape[:-1], 1), dtype=bool)
  is_2x2_small = is_neg & jnp.concatenate([is_neg[..., 1:], pad_end], axis=-1)

  k_range = jnp.arange(n)
  if lower:
    is_2x2_first = is_2x2_small
    is_2x2_second = jnp.concatenate([pad_beg, is_2x2_first[..., :-1]], axis=-1)
    swap_a = jnp.where(is_2x2_first, jnp.minimum(k_range + 1, n - 1), k_range)
  else:
    is_2x2_first = is_neg & jnp.concatenate([pad_beg, is_neg[..., :-1]], axis=-1)
    is_2x2_second = jnp.concatenate([is_2x2_first[..., 1:], pad_end], axis=-1)
    swap_a = jnp.where(is_2x2_first, jnp.maximum(k_range - 1, 0), k_range)

  swap_b = jnp.abs(ipiv) - 1
  is_noop = is_2x2_second | (swap_a == swap_b)
  swap_a = jnp.where(is_noop, k_range, swap_a)
  swap_b = jnp.where(is_noop, k_range, swap_b)

  # Build perms[..., j, k]: permutation vector for BK step k, applied to row index j.
  # Shape (..., n, n): column k = identity with swap_a[k] ↔ swap_b[k].
  j_idx = jnp.arange(n)
  perms = jnp.where(
    j_idx[:, None] == swap_a[..., None, :],
    swap_b[..., None, :],
    jnp.where(
      j_idx[:, None] == swap_b[..., None, :],
      swap_a[..., None, :],
      j_idx[:, None],
    ),
  ).astype(jnp.int32)  # (..., n, n)

  # Shift perms by one column so column k holds P(k+1)'s swap (lower) or
  # P(k-1)'s swap (upper).  Identity column broadcast to (..., n, 1).
  id_col = jnp.broadcast_to(
    j_idx.reshape(*([1] * (perms.ndim - 2)), n, 1), (*perms.shape[:-1], 1)
  )
  if lower:
    perms_shifted = jnp.concatenate([perms[..., 1:], id_col], axis=-1)
  else:
    perms_shifted = jnp.concatenate([id_col, perms[..., :-1]], axis=-1)

  # Use explicit non-negative axis — lax.rev (used by associative_scan) rejects -1.
  col_axis = perms.ndim - 1  # last axis (step/column axis), always >= 0

  P_correct = jax.lax.associative_scan(
    _bk_perm_compose, perms_shifted, reverse=lower, axis=col_axis
  )

  # Apply P_correct directly to factors: D entries are preserved (proved above),
  # strict triangular region gets the corrected L multipliers.
  return jnp.take_along_axis(factors, P_correct, axis=-2)


def _ldl_d_solve(factors, ipiv, y, lower, hermitian):
  """Block-diagonal D^{-1} solve. y has shape [..., n, nrhs]."""
  diag = jnp.diagonal(factors, axis1=-2, axis2=-1)  # [..., n]
  offset = -1 if lower else +1
  off = jnp.diagonal(factors, offset=offset, axis1=-2, axis2=-1)  # [..., n-1]
  # Pad off-diagonal to length n (boundary element is always a 1×1 block)
  sub_n = jnp.concatenate(
      [off, jnp.zeros((*off.shape[:-1], 1), dtype=off.dtype)], axis=-1)  # [..., n]

  is_neg = ipiv < 0
  pad = jnp.zeros((*is_neg.shape[:-1], 1), dtype=bool)
  is_2x2_start = is_neg & jnp.concatenate([is_neg[..., 1:], pad], axis=-1)
  is_2x2_end = jnp.concatenate([pad, is_2x2_start[..., :-1]], axis=-1)

  diag_next = jnp.roll(diag, -1, axis=-1)   # D[k+1,k+1] from k's perspective
  diag_prev = jnp.roll(diag, +1, axis=-1)   # D[k-1,k-1] from k's perspective
  sub_prev = jnp.roll(sub_n, +1, axis=-1)   # sub_n[k-1] shifted to position k

  y_next = jnp.roll(y, -1, axis=-2)
  y_prev = jnp.roll(y, +1, axis=-2)

  # det for 2×2 block starting at k: a*c - |b|^2 (hermitian) or a*c - b^2
  if hermitian:
    det = diag * diag_next - sub_n * jnp.conj(sub_n)
    if lower:
      # sub_n[k] = D[k+1,k] (subdiagonal). The block is [[a, conj(b)],[b, c]].
      # z_start row k needs -conj(b) = -conj(sub_n[k]).
      # z_end row k+1 needs -b = -sub_n[k], so use sub_prev (no conj).
      sub_n_start = jnp.conj(sub_n)
      conj_sub_prev = sub_prev
    else:
      # sub_n[k] = D[k,k+1] (superdiagonal). The block is [[a, b],[conj(b), c]].
      # z_start row k needs -b = -sub_n[k] (no conj).
      # z_end row k+1 needs -conj(b) = -conj(sub_n[k]).
      sub_n_start = sub_n
      conj_sub_prev = jnp.conj(sub_prev)
  else:
    det = diag * diag_next - sub_n ** 2
    sub_n_start = sub_n
    conj_sub_prev = sub_prev

  det_prev = jnp.roll(det, +1, axis=-1)

  z_start = (diag_next[..., None] * y - sub_n_start[..., None] * y_next) / det[..., None]
  z_end = (diag_prev[..., None] * y - conj_sub_prev[..., None] * y_prev) / det_prev[..., None]
  z_1x1 = y / diag[..., None]

  return jnp.where(is_2x2_start[..., None], z_start,
                   jnp.where(is_2x2_end[..., None], z_end, z_1x1))


def _ldl_solve_core(factors, ipiv, perm, b, *, lower, hermitian):
  """Pure-JAX LDL solve: P b → (L or U)⁻¹ → D⁻¹ → (L or U)^{-T/H} → P^T x.

  Both lower=True  (P A P^T = L D L^{T/H}) and
       lower=False (P A P^T = U D U^{T/H}) use the same triangular-solve order:
  apply the non-transposed factor first, then D^{-1}, then the transposed factor.

  Works on CPU and GPU (dispatches to cuBLAS triangular solves on GPU).
  Uses take_along_axis for batched-safe permutation gather.

  ``factors`` is ``factors_untangled``: L_correct in the strict triangular
  region, D on the diagonal, and D off-diagonal entries at ±1 (for 2×2
  blocks).  _ldl_d_solve reads D directly from ``factors``.  triangular_solve
  must not see the D off-diagonal entries as L multipliers, so we zero them
  first.
  """
  n = factors.shape[-1]
  is_neg = ipiv < 0
  pad = jnp.zeros((*is_neg.shape[:-1], 1), dtype=bool)
  is_2x2 = is_neg & jnp.concatenate([is_neg[..., 1:], pad], axis=-1)
  ij = jnp.arange(n)
  if lower:
    d_off_mask = (ij[:, None] == ij[None, :] + 1) & is_2x2[..., None, :]
    L = jnp.where(d_off_mask, 0, jnp.tril(factors, k=-1))
  else:
    d_off_mask = (ij[:, None] == ij[None, :] - 1) & is_2x2[..., :, None]
    L = jnp.where(d_off_mask, 0, jnp.triu(factors, k=+1))

  x = jnp.take_along_axis(b, perm[..., None], axis=-2)           # P b
  x = triangular_solve(L, x, left_side=True, lower=lower,
                        unit_diagonal=True)                       # (L or U)⁻¹
  x = _ldl_d_solve(factors, ipiv, x, lower, hermitian)           # D⁻¹
  x = triangular_solve(L, x, left_side=True, lower=lower,
                        unit_diagonal=True, transpose_a=True,
                        conjugate_a=hermitian)                    # (L or U)^{-T/H}
  inv_perm = jnp.argsort(perm, axis=-1).astype(jnp.int32)
  return jnp.take_along_axis(x, inv_perm[..., None], axis=-2)    # P^T x


def _ldl_solve_batching_rule(batched_args, batch_dims, *, lower, hermitian):
  factors, ipiv, perm, b = batched_args
  f_bd, i_bd, p_bd, b_bd = batch_dims

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
  perm = (
    batching.moveaxis(perm, p_bd, 0)
    if p_bd is not None
    else jnp.expand_dims(perm, 0)
  )
  b = (
    batching.moveaxis(b, b_bd, 0)
    if b_bd is not None
    else jnp.expand_dims(b, 0)
  )

  x = ldl_solve_p.bind(factors, ipiv, perm, b, lower=lower, hermitian=hermitian)
  return x, 0


ldl_solve_p = core.Primitive("ldl_solve")
ldl_solve_p.multiple_results = False
ldl_solve_p.def_abstract_eval(_ldl_solve_abstract_eval)
ldl_solve_p.def_impl(functools.partial(dispatch.apply_primitive, ldl_solve_p))
mlir.register_lowering(
  ldl_solve_p,
  mlir.lower_fun(_ldl_solve_core, multiple_results=False),
)
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
