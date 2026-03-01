"""
jaxtra._core — ormqr primitive, verbatim from JAX PR #35104.

The primitive registration (standard_linalg_primitive, lowerings) is copied
verbatim from jax/_src/lax/linalg.py (PR #35104).  The LAPACK FFI targets
(lapack_sormqr_ffi, lapack_dormqr_ffi, lapack_cunmqr_ffi, lapack_zunmqr_ffi)
are registered from our C extension, which uses the same names as jaxlib will
once the PR merges.  Once it does, this module can be replaced by a straight
import from jax._src.lax.linalg.
"""
from __future__ import annotations

import os
import glob
import importlib.util
import sys

import numpy as np

from jax._src import core, dtypes
from jax._src.lax import lax
from jax._src.lax import control_flow
from jax._src.interpreters import mlir
from jax._src.lax.linalg import (
    standard_linalg_primitive,
    register_cpu_gpu_lowering,
    _linalg_ffi_lowering,
    _float,
    _complex,
    _tril,
)
from jaxlib import lapack
from jax._src.typing import ArrayLike, Array


# ---------------------------------------------------------------------------
# Register FFI targets from the C extension
# (same target names as jaxlib will expose once PR #35104 merges)
# ---------------------------------------------------------------------------

def _load_extension():
    mod_name = "jaxtra._jaxtra"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    here = os.path.dirname(__file__)
    candidates = []
    for d in (here, os.path.dirname(here)):
        candidates.extend(glob.glob(os.path.join(d, "_jaxtra*.so")))
    if not candidates:
        raise ImportError(
            "jaxtra C extension (_jaxtra.so) not found. "
            "Build it with:  pip install -e . --no-build-isolation"
        )
    spec = importlib.util.spec_from_file_location(mod_name, candidates[0])
    ext = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = ext
    spec.loader.exec_module(ext)
    return ext


import jax.ffi as ffi

_jaxtra = _load_extension()
_jaxtra.initialize()
for _platform, _targets in _jaxtra.registrations().items():
    for _name, _capsule, _api_version in _targets:
        ffi.register_ffi_target(
            _name, _capsule, platform=_platform, api_version=_api_version
        )


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
