"""Tests for jaxtra — XLA FFI LAPACK ORMQR kernel.

Covers:
  • All four operation modes (left/right × transpose/no-transpose)
  • All four supported dtypes (float32, float64, complex64, complex128)
  • Batched inputs
  • JIT compilation
  • vmap (vectorised map)
  • jaxtra.scipy.linalg.qr_multiply
"""

import numpy as np
import pytest
import scipy.linalg as scipy_linalg
import jax
import jax.numpy as jnp

# Enable 64-bit floats in JAX (required for float64 / complex128 tests).
jax.config.update("jax_enable_x64", True)

from jax._src.lax.linalg import geqrf   # JAX's existing geqrf primitive
from jaxtra.lax.linalg import ormqr     # our new XLA FFI primitive
from jaxtra.scipy.linalg import qr_multiply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

REAL_DTYPES    = [np.float32, np.float64]
COMPLEX_DTYPES = [np.complex64, np.complex128]
ALL_DTYPES     = REAL_DTYPES + COMPLEX_DTYPES
SHAPES         = [(4, 3), (5, 5), (6, 2)]


def _random(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)).astype(dtype)
    return RNG.standard_normal(shape).astype(dtype)


def _geqrf_jax(a_np):
    """Run JAX geqrf and return numpy arrays (H, taus)."""
    H, taus = geqrf(jnp.array(a_np))
    return np.array(H), np.array(taus)


def _geqrf_scipy(a):
    """Run scipy geqrf and return (H, taus)."""
    import scipy.linalg.lapack as sl
    dispatch = {
        np.dtype("float32"):    sl.sgeqrf,
        np.dtype("float64"):    sl.dgeqrf,
        np.dtype("complex64"):  sl.cgeqrf,
        np.dtype("complex128"): sl.zgeqrf,
    }
    r, tau, _, info = dispatch[np.dtype(a.dtype)](a)
    assert info == 0
    return r, tau


def _ref_ormqr(side, trans, H, tau, C):
    """Reference using scipy's LAPACK wrapper."""
    import scipy.linalg.lapack as sl
    dispatch = {
        np.dtype("float32"):    sl.sormqr,
        np.dtype("float64"):    sl.dormqr,
        np.dtype("complex64"):  sl.cunmqr,
        np.dtype("complex128"): sl.zunmqr,
    }
    fn = dispatch[np.dtype(H.dtype)]
    lwork = int(np.real(fn(side, trans, H, tau, C, lwork=-1)[1][0]))
    return fn(side, trans, H, tau, C, lwork=lwork)[0]


def _trans_char(dtype, transpose: bool) -> str:
    """TRANS character for LAPACK: 'C' for complex, 'T' for real, 'N' for no-op."""
    if not transpose:
        return 'N'
    return 'C' if np.issubdtype(dtype, np.complexfloating) else 'T'


def _run_ormqr(a_np, c_np, *, left, transpose):
    """Run jaxtra.ormqr and scipy reference; return (result, reference)."""
    H_jax, taus_jax = _geqrf_jax(a_np)
    H_sc, taus_sc   = _geqrf_scipy(a_np)

    result = np.array(ormqr(
        jnp.array(H_jax), jnp.array(taus_jax), jnp.array(c_np),
        left=left, transpose=transpose,
    ))
    side  = 'L' if left else 'R'
    trans = _trans_char(a_np.dtype, transpose)
    ref = _ref_ormqr(side, trans, H_sc, taus_sc, c_np)
    return result, ref


# ---------------------------------------------------------------------------
# ormqr tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ormqr_left_notranspose(dtype, shape):
    m, n = shape
    a = _random((m, n), dtype)
    c = _random((m, m), dtype)  # full-Q acts on m rows
    result, ref = _run_ormqr(a, c, left=True, transpose=False)
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ormqr_left_transpose(dtype, shape):
    m, n = shape
    a = _random((m, n), dtype)
    c = _random((m, m), dtype)
    result, ref = _run_ormqr(a, c, left=True, transpose=True)
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ormqr_right_notranspose(dtype, shape):
    m, n = shape
    a = _random((m, n), dtype)
    c = _random((n, m), dtype)  # C @ Q: C cols must match Q's row size
    result, ref = _run_ormqr(a, c, left=False, transpose=False)
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_ormqr_right_transpose(dtype, shape):
    m, n = shape
    a = _random((m, n), dtype)
    c = _random((n, m), dtype)
    result, ref = _run_ormqr(a, c, left=False, transpose=True)
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_ormqr_batch(dtype):
    batch, m, k = 3, 5, 3
    a_batch = _random((batch, m, k), dtype)
    c_batch = _random((batch, m, m), dtype)

    # Reference: loop over batch.
    H_sc_batch  = np.stack([_geqrf_scipy(a_batch[i])[0] for i in range(batch)])
    tau_sc_batch = np.stack([_geqrf_scipy(a_batch[i])[1] for i in range(batch)])
    ref = np.stack([
        _ref_ormqr('L', 'N', H_sc_batch[i], tau_sc_batch[i], c_batch[i])
        for i in range(batch)
    ])

    H_jax  = jnp.array(np.stack([_geqrf_jax(a_batch[i])[0] for i in range(batch)]))
    tau_jax = jnp.array(np.stack([_geqrf_jax(a_batch[i])[1] for i in range(batch)]))
    result = np.array(ormqr(H_jax, tau_jax, jnp.array(c_batch),
                             left=True, transpose=False))
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


def test_ormqr_jit():
    dtype = np.float64
    m, k = 6, 3
    a = _random((m, k), dtype)
    c = _random((m, m), dtype)
    H_np, taus_np = _geqrf_jax(a)

    @jax.jit
    def fn(H, taus, c):
        return ormqr(H, taus, c, left=True, transpose=True)

    result = np.array(fn(jnp.array(H_np), jnp.array(taus_np), jnp.array(c)))
    H_sc, taus_sc = _geqrf_scipy(a)
    ref = _ref_ormqr('L', 'T', H_sc, taus_sc, c)
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


def test_ormqr_vmap():
    dtype = np.float64
    batch, m, k = 4, 5, 3
    a_batch = _random((batch, m, k), dtype)
    c_batch = _random((batch, m, m), dtype)

    @jax.vmap
    def fn(a, c):
        H, taus = geqrf(a)
        return ormqr(H, taus, c, left=True, transpose=False)

    result = np.array(fn(jnp.array(a_batch), jnp.array(c_batch)))
    assert result.shape == (batch, m, m)

    # Validate each batch element.
    for i in range(batch):
        H_sc, taus_sc = _geqrf_scipy(a_batch[i])
        ref = _ref_ormqr('L', 'N', H_sc, taus_sc, c_batch[i])
        np.testing.assert_allclose(result[i], ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# qr_multiply tests  (scipy-compatible interface)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_qr_multiply_left(dtype):
    """mode='left' computes Q_thin @ c where Q_thin is m×k (thin Q)."""
    m, n = 5, 3
    k = min(m, n)
    a = _random((m, n), dtype)
    c = _random((k, 4), dtype)    # c must have k = min(m,n) rows
    result, R = qr_multiply(a, c, mode='left')
    Q_thin, _ = scipy_linalg.qr(a, mode='economic')
    ref = Q_thin @ c
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_qr_multiply_right(dtype):
    """mode='right' computes c @ Q_thin where Q_thin is m×k (thin Q)."""
    m, n = 5, 3
    a = _random((m, n), dtype)
    c = _random((4, m), dtype)    # c must have m cols
    result, R = qr_multiply(a, c, mode='right')
    Q_thin, _ = scipy_linalg.qr(a, mode='economic')
    ref = c @ Q_thin
    np.testing.assert_allclose(result, ref, rtol=1e-4, atol=1e-5)


def test_qr_multiply_least_squares():
    """ormqr-based least-squares solve matches numpy.linalg.lstsq."""
    m, n = 8, 4
    A = _random((m, n), np.float64)
    b = _random((m,), np.float64)

    # Standard numpy least-squares reference.
    x_ref, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # ormqr-based solve: form Qᵀb then back-substitute with R.
    H_sc, taus_sc, _, _ = scipy_linalg.lapack.dgeqrf(A)
    lw = int(scipy_linalg.lapack.dormqr('L', 'T', H_sc, taus_sc, b[:, None], lwork=-1)[1][0])
    Qtb = scipy_linalg.lapack.dormqr('L', 'T', H_sc, taus_sc, b[:, None], lwork=lw)[0]
    R = np.triu(H_sc[:n, :n])
    x_ormqr = np.linalg.solve(R, Qtb[:n])

    np.testing.assert_allclose(x_ormqr.ravel(), x_ref, rtol=1e-8)
