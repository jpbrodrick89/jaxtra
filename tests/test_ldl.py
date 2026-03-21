"""Tests for jaxtra — ldl primitive and jaxtra.scipy.linalg.ldl.

References are computed from JAX operations (not numpy/scipy), following the
convention in CLAUDE.md.  The correctness check is:

    lu @ d @ lu^{T/H} ≈ a[perm][:, perm]

where (lu, d, perm) come from jaxtra.scipy.linalg.ldl.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import ldl as ldl_primitive, ldl_solve
from jaxtra.scipy.linalg import ldl

RNG = np.random.default_rng(1234)

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
all_dtypes = float_types + complex_types


def _rand_sym(n, dtype, batch=()):
  """Random symmetric matrix (positive diagonal to avoid pure singularity)."""
  shape = batch + (n, n)
  if np.issubdtype(dtype, np.complexfloating):
    a = (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)).astype(
      dtype
    )
    a = a + np.conj(a).swapaxes(-1, -2)  # Hermitian
  else:
    a = RNG.standard_normal(shape).astype(dtype)
    a = a + a.swapaxes(-1, -2)  # symmetric
  # Add diagonal to ensure the matrix is properly conditioned (not all-diagonal
  # zeros). We use alternating signs so the matrix is indefinite.
  diag = np.zeros(shape[:-1], dtype=dtype)
  diag[..., ::2] = 5
  diag[..., 1::2] = -5
  a += np.einsum("...i,...ij->...ij", diag, np.eye(n, dtype=dtype))
  return a


def _tol(dtype):
  return {
    np.float32: 1e-4,
    np.complex64: 1e-4,
    np.float64: 1e-10,
    np.complex128: 1e-10,
  }[dtype]


# ---------------------------------------------------------------------------
# ldl primitive — returns (factors, ipiv)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", [4, 7])
def test_ldl_primitive_shapes(n, dtype, lower):
  a = jnp.array(_rand_sym(n, dtype))
  hermitian = np.issubdtype(dtype, np.complexfloating)
  factors, ipiv, perm = ldl_primitive(a, lower=lower, hermitian=hermitian)
  assert factors.shape == (n, n)
  assert factors.dtype == a.dtype
  assert ipiv.shape == (n,)
  assert ipiv.dtype == jnp.int32
  assert perm.shape == (n,)
  assert perm.dtype == jnp.int32


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", [4, 7])
def test_ldl_primitive_batched(n, dtype, lower):
  a = jnp.array(_rand_sym(n, dtype, batch=(3,)))
  hermitian = np.issubdtype(dtype, np.complexfloating)
  factors, ipiv, perm = ldl_primitive(a, lower=lower, hermitian=hermitian)
  assert factors.shape == (3, n, n)
  assert ipiv.shape == (3, n)
  assert perm.shape == (3, n)


# ---------------------------------------------------------------------------
# jaxtra.scipy.linalg.ldl — returns (lu, d, perm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", [3, 5, 8])
def test_ldl_correctness(n, dtype, lower):
  """lu @ d @ lu^{T/H} == a[perm][:, perm]."""
  a = jnp.array(_rand_sym(n, dtype))
  hermitian = np.issubdtype(dtype, np.complexfloating)
  lu, d, perm = ldl(a, lower=lower, hermitian=hermitian)

  # Build permuted A.
  a_perm = a[perm][:, perm]

  # Reconstruct: lu @ d @ lu^{T/H}.
  # Use "highest" precision to disable tensor cores (which use FP16 internally
  # on A100/H100 for float32/complex64 and accumulate too much error).
  if hermitian:
    lu_T = jnp.conj(lu).swapaxes(-1, -2)
  else:
    lu_T = lu.swapaxes(-1, -2)
  with jax.default_matmul_precision("highest"):
    reconstructed = lu @ d @ lu_T

  tol = _tol(dtype)
  np.testing.assert_allclose(reconstructed, a_perm, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", [3, 6])
def test_ldl_batched_correctness(n, dtype, lower):
  """Batched ldl: check each element independently."""
  batch = 2
  a_batch = jnp.array(_rand_sym(n, dtype, batch=(batch,)))
  hermitian = np.issubdtype(dtype, np.complexfloating)
  lu_batch, d_batch, perm_batch = ldl(a_batch, lower=lower, hermitian=hermitian)

  tol = _tol(dtype)
  for i in range(batch):
    a_perm = a_batch[i][perm_batch[i]][:, perm_batch[i]]
    if hermitian:
      lu_T = jnp.conj(lu_batch[i]).swapaxes(-1, -2)
    else:
      lu_T = lu_batch[i].swapaxes(-1, -2)
    with jax.default_matmul_precision("highest"):
      reconstructed = lu_batch[i] @ d_batch[i] @ lu_T
    np.testing.assert_allclose(reconstructed, a_perm, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# JIT and vmap over ldl primitive
# ---------------------------------------------------------------------------


def test_ldl_primitive_jit():
  n = 5
  a = jnp.array(_rand_sym(n, np.float64))

  @jax.jit
  def fn(a):
    return ldl_primitive(a, lower=True, hermitian=False)

  factors_ref, ipiv_ref, perm_ref = ldl_primitive(a, lower=True, hermitian=False)
  factors_jit, ipiv_jit, perm_jit = fn(a)

  np.testing.assert_allclose(factors_jit, factors_ref, rtol=1e-10, atol=1e-10)
  np.testing.assert_array_equal(ipiv_jit, ipiv_ref)
  np.testing.assert_array_equal(perm_jit, perm_ref)


def test_ldl_primitive_vmap():
  batch, n = 4, 5
  a_batch = jnp.array(_rand_sym(n, np.float64, batch=(batch,)))

  @jax.vmap
  def fn(a):
    return ldl_primitive(a, lower=True, hermitian=False)

  factors_vmap, ipiv_vmap, perm_vmap = fn(a_batch)
  assert factors_vmap.shape == (batch, n, n)
  assert ipiv_vmap.shape == (batch, n)
  assert perm_vmap.shape == (batch, n)

  # Check each slice matches the unbatched result.
  for i in range(batch):
    factors_i, ipiv_i, perm_i = ldl_primitive(a_batch[i], lower=True, hermitian=False)
    np.testing.assert_allclose(
      factors_vmap[i], factors_i, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_array_equal(ipiv_vmap[i], ipiv_i)
    np.testing.assert_array_equal(perm_vmap[i], perm_i)


# ---------------------------------------------------------------------------
# ldl_solve correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("n", [4, 6])
def test_ldl_solve_correctness(n, dtype, lower):
  """ldl_solve(factors, ipiv, perm, b) ≈ A^{-1} @ b."""
  a = jnp.array(_rand_sym(n, dtype))
  b = jnp.array(RNG.standard_normal((n, 2)).astype(dtype))
  hermitian = np.issubdtype(dtype, np.complexfloating)

  factors, ipiv, perm = ldl_primitive(a, lower=lower, hermitian=hermitian)
  x = ldl_solve(factors, ipiv, perm, b, lower=lower, hermitian=hermitian)

  tol = _tol(dtype)
  with jax.default_matmul_precision("highest"):
    np.testing.assert_allclose(a @ x, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("n", [4, 6])
def test_ldl_solve_1d_rhs(n, dtype):
  """ldl_solve with 1-D b returns 1-D x."""
  a = jnp.array(_rand_sym(n, dtype))
  b = jnp.array(RNG.standard_normal((n,)).astype(dtype))
  hermitian = np.issubdtype(dtype, np.complexfloating)

  factors, ipiv, perm = ldl_primitive(a, lower=True, hermitian=hermitian)
  x = ldl_solve(factors, ipiv, perm, b, lower=True, hermitian=hermitian)

  assert x.shape == (n,)
  tol = _tol(dtype)
  with jax.default_matmul_precision("highest"):
    np.testing.assert_allclose(a @ x, b, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_ldl_solve_jit(dtype):
  """ldl_solve is JIT-compatible."""
  n = 5
  a = jnp.array(_rand_sym(n, dtype))
  b = jnp.array(RNG.standard_normal((n,)).astype(dtype))
  hermitian = np.issubdtype(dtype, np.complexfloating)

  factors, ipiv, perm = ldl_primitive(a, lower=True, hermitian=hermitian)

  @jax.jit
  def fn(factors, ipiv, perm, b):
    return ldl_solve(factors, ipiv, perm, b, lower=True, hermitian=hermitian)

  x_jit = fn(factors, ipiv, perm, b)
  x_ref = ldl_solve(factors, ipiv, perm, b, lower=True, hermitian=hermitian)

  tol = _tol(dtype)
  np.testing.assert_allclose(x_jit, x_ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_ldl_solve_vmap(dtype):
  """ldl_solve is vmap-compatible."""
  batch, n = 3, 5
  a_batch = jnp.array(_rand_sym(n, dtype, batch=(batch,)))
  b_batch = jnp.array(RNG.standard_normal((batch, n)).astype(dtype))
  hermitian = np.issubdtype(dtype, np.complexfloating)

  factors_batch, ipiv_batch, perm_batch = ldl_primitive(
    a_batch, lower=True, hermitian=hermitian
  )

  @jax.vmap
  def fn(factors, ipiv, perm, b):
    return ldl_solve(factors, ipiv, perm, b, lower=True, hermitian=hermitian)

  x_vmap = fn(factors_batch, ipiv_batch, perm_batch, b_batch)
  assert x_vmap.shape == (batch, n)

  tol = _tol(dtype)
  for i in range(batch):
    with jax.default_matmul_precision("highest"):
      np.testing.assert_allclose(
        a_batch[i] @ x_vmap[i], b_batch[i], rtol=tol, atol=tol
      )


# ---------------------------------------------------------------------------
# Symmetric (non-Hermitian) complex test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("n", [4, 6])
def test_ldl_complex_symmetric(n, dtype):
  """Complex symmetric (hermitian=False) ldl correctness."""
  # Build a complex symmetric (not Hermitian) matrix: a == a.T (not conj).
  a_np = RNG.standard_normal((n, n)).astype(dtype) + 1j * RNG.standard_normal((
    n,
    n,
  )).astype(np.float32 if dtype == np.complex64 else np.float64)
  a_np = a_np + a_np.T  # symmetric, not Hermitian
  # Add conditioning.
  a_np += np.diag(np.where(np.arange(n) % 2 == 0, 10, -10).astype(dtype))
  a = jnp.array(a_np)

  lu, d, perm = ldl(a, lower=True, hermitian=False)

  a_perm = a[perm][:, perm]
  with jax.default_matmul_precision("highest"):
    reconstructed = lu @ d @ lu.swapaxes(-1, -2)  # lu @ d @ lu^T (not conj)

  tol = _tol(dtype)
  np.testing.assert_allclose(reconstructed, a_perm, rtol=tol, atol=tol)
