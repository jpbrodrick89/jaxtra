"""Tests for jaxtra.lax.linalg.ormqr and jaxtra.scipy.linalg.qr_multiply.

Mirrors the test patterns from the upstream JAX PR #35104.
"""

import numpy as np
import pytest
import scipy.linalg as scipy_linalg

from jaxtra.lax.linalg import ormqr
from jaxtra.scipy.linalg import qr_multiply


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        real = RNG.standard_normal(shape)
        imag = RNG.standard_normal(shape)
        return (real + 1j * imag).astype(dtype)
    return RNG.standard_normal(shape).astype(dtype)


def _geqrf(a):
    """Return (householder_matrix, tau) using scipy."""
    import scipy.linalg.lapack as sl
    dtype = a.dtype
    if dtype == np.float32:
        r, tau, _, info = sl.sgeqrf(a)
    elif dtype == np.float64:
        r, tau, _, info = sl.dgeqrf(a)
    elif dtype == np.complex64:
        r, tau, _, info = sl.cgeqrf(a)
    else:
        r, tau, _, info = sl.zgeqrf(a)
    assert info == 0
    return r, tau


def _form_q(r, tau):
    """Materialise Q from (r, tau) using scipy."""
    m, n = r.shape
    k = tau.shape[0]
    return scipy_linalg.qr(r, mode='economic')[0]


# ---------------------------------------------------------------------------
# ormqr tests
# ---------------------------------------------------------------------------

DTYPES = [np.float32, np.float64, np.complex64, np.complex128]
SHAPES = [(4, 3), (5, 5), (6, 2)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_ormqr_left_notranspose(m, n, dtype):
    """ormqr with left=True, transpose=False computes Q @ C (full Q, m×m)."""
    a = _random((m, n), dtype)
    c = _random((m, 2), dtype)

    r, tau = _geqrf(a.copy())
    result = ormqr(r, tau, c.copy(), left=True, transpose=False)

    # LAPACK dormqr uses the full m×m Q, not the economy Q.
    Q_full, _ = scipy_linalg.qr(a, mode='full')
    ref = Q_full @ c
    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_ormqr_left_transpose(m, n, dtype):
    """ormqr with left=True, transpose=True computes Q^H @ C (full Q, m×m)."""
    a = _random((m, n), dtype)
    c = _random((m, 2), dtype)

    r, tau = _geqrf(a.copy())
    result = ormqr(r, tau, c.copy(), left=True, transpose=True)

    # Reference: full Q^H @ c.
    Q_full, _ = scipy_linalg.qr(a, mode='full')
    ref = Q_full.conj().T @ c
    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_ormqr_right_notranspose(m, n, dtype):
    """ormqr with left=False, transpose=False computes C @ Q."""
    a = _random((m, n), dtype)
    k = min(m, n)
    c = _random((2, m), dtype)

    r, tau = _geqrf(a.copy())
    result = ormqr(r, tau, c.copy(), left=False, transpose=False)

    ref, _ = scipy_linalg.qr_multiply(a, c, mode='right')
    # scipy qr_multiply with mode='right' gives c @ Q[:, :k], so we need to
    # compare only the first k columns of result.
    np.testing.assert_allclose(result[:, :k], ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_ormqr_right_transpose(m, n, dtype):
    """ormqr with left=False, transpose=True computes C @ Q^H."""
    a = _random((m, n), dtype)
    c = _random((2, m), dtype)

    r, tau = _geqrf(a.copy())
    result = ormqr(r, tau, c.copy(), left=False, transpose=True)

    # Reference via explicit Q
    Q, _ = scipy_linalg.qr(a)  # Q is m×m
    ref = c @ Q.conj().T
    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ormqr_batch(dtype):
    """ormqr handles a batch dimension."""
    batch = 3
    m, n = 5, 3
    a = _random((batch, m, n), dtype)
    c = _random((batch, m, 2), dtype)

    r = np.empty_like(a)
    tau = np.empty((batch, n), dtype=dtype)
    for i in range(batch):
        r[i], tau[i] = _geqrf(a[i].copy())

    result = ormqr(r, tau, c.copy(), left=True, transpose=True)

    for i in range(batch):
        Q_full, _ = scipy_linalg.qr(a[i], mode='full')
        ref = Q_full.conj().T @ c[i]
        np.testing.assert_allclose(result[i], ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# qr_multiply tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_qr_multiply_left(m, n, dtype):
    """qr_multiply mode='left' computes Q @ c."""
    a = _random((m, n), dtype)
    k = min(m, n)
    c = _random((k, 2), dtype)

    result, R = qr_multiply(a, c, mode='left')
    ref, R_ref = scipy_linalg.qr_multiply(a, c, mode='left')

    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(R, R_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,n", SHAPES)
def test_qr_multiply_right(m, n, dtype):
    """qr_multiply mode='right' computes c @ Q."""
    a = _random((m, n), dtype)
    c = _random((2, m), dtype)

    result, R = qr_multiply(a, c, mode='right')
    ref, R_ref = scipy_linalg.qr_multiply(a, c, mode='right')

    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(R, R_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_qr_multiply_1d_right(dtype):
    """qr_multiply handles a 1-D c vector with mode='right'."""
    m, n = 5, 3
    a = _random((m, n), dtype)
    c = _random((m,), dtype)

    result, R = qr_multiply(a, c, mode='right')
    ref, R_ref = scipy_linalg.qr_multiply(a, c, mode='right')

    assert result.ndim == 1
    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_qr_multiply_1d_left(dtype):
    """qr_multiply handles a 1-D c vector with mode='left'."""
    m, n = 5, 3
    k = min(m, n)
    a = _random((m, n), dtype)
    c = _random((k,), dtype)

    result, R = qr_multiply(a, c, mode='left')
    ref, R_ref = scipy_linalg.qr_multiply(a, c, mode='left')

    assert result.ndim == 1
    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_qr_multiply_pivoting(dtype):
    """qr_multiply with pivoting=True returns a permutation array."""
    m, n = 5, 3
    a = _random((m, n), dtype)
    c = _random((m,), dtype)

    result, R, P = qr_multiply(a, c, mode='right', pivoting=True)
    ref, R_ref, P_ref = scipy_linalg.qr_multiply(a, c, mode='right', pivoting=True)

    np.testing.assert_array_equal(P, P_ref)
    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qr_multiply_least_squares(dtype):
    """qr_multiply can be used to solve a least-squares problem."""
    m, n = 6, 3
    A = _random((m, n), dtype)
    b = _random((m,), dtype)

    Qtb, R = qr_multiply(A, b, mode='right')

    # x = R^{-1} @ Q^T @ b  (R is k×n = n×n here since m≥n)
    x = scipy_linalg.solve_triangular(R, Qtb)

    # Reference via numpy lstsq
    x_ref, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(x, x_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_qr_multiply_conjugate(dtype):
    """qr_multiply conjugate=True uses conj(Q)."""
    m, n = 5, 3
    a = _random((m, n), dtype)
    c = _random((2, m), dtype)

    result, R = qr_multiply(a, c, mode='right', conjugate=True)
    ref, R_ref = scipy_linalg.qr_multiply(a, c, mode='right', conjugate=True)

    np.testing.assert_allclose(result, ref, atol=1e-5, rtol=1e-5)


def test_qr_multiply_invalid_mode():
    a = np.eye(3)
    c = np.eye(3)
    with pytest.raises(ValueError, match="mode must be"):
        qr_multiply(a, c, mode='invalid')


def test_ormqr_shape_mismatch_left():
    m, n = 4, 3
    a = _random((m, n), np.float64)
    c = _random((n, 2), np.float64)  # wrong: should be (m, 2)
    r, tau = _geqrf(a)
    with pytest.raises(ValueError, match="left=True"):
        ormqr(r, tau, c, left=True, transpose=False)


def test_ormqr_shape_mismatch_right():
    m, n = 4, 3
    a = _random((m, n), np.float64)
    c = _random((2, n), np.float64)  # wrong: should be (2, m)
    r, tau = _geqrf(a)
    with pytest.raises(ValueError, match="left=False"):
        ormqr(r, tau, c, left=False, transpose=False)
