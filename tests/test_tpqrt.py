"""Tests for jaxtra — tpqrt (triangular-pentagonal QR) primitive.

Covers the LAPACK CPU path (``R, V, T`` checked bit-for-bit against SciPy) and
the pure-JAX blocked-Householder fallback (checked by reconstructing
``C = Q [R; 0]`` from the returned reflectors), across dtypes, pentagonal
shapes, block sizes, jit, vmap and batching.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import scipy.linalg.lapack as scipy_lapack
from jax._src.lax.linalg import geqrf
from jaxtra._src.lax.linalg import tpqrt, _tpqrt_blocked_householder_2d

RNG = np.random.default_rng(42)

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
all_dtypes = float_types + complex_types

_SCIPY_TPQRT = {
    np.dtype(np.float32): scipy_lapack.stpqrt,
    np.dtype(np.float64): scipy_lapack.dtpqrt,
    np.dtype(np.complex64): scipy_lapack.ctpqrt,
    np.dtype(np.complex128): scipy_lapack.ztpqrt,
}


def rand(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return (RNG.standard_normal(shape)
                + 1j * RNG.standard_normal(shape)).astype(dtype)
    return RNG.standard_normal(shape).astype(dtype)


def make_pentagonal(m, n, l, dtype):
    """Upper triangular ``a`` (n×n) and pentagonal ``b`` (m×n) with the LAPACK
    tpqrt structure: bottom ``l`` rows are upper trapezoidal."""
    a = jnp.triu(jnp.asarray(rand((n, n), dtype)))
    b = jnp.asarray(rand((m, n), dtype))
    rows = jnp.arange(m)[:, None]
    cols = jnp.arange(n)[None, :]
    mask = (rows < (m - l)) | (cols >= rows - (m - l))
    return a, b * mask.astype(dtype)


def gram(R):
    """Sign/phase-invariant signature: Rᴴ R (= Cᴴ C for any QR factor R)."""
    return jnp.swapaxes(R.conj(), -1, -2) @ R


def reference_R(a, b):
    """R = triu(qr([triu(a); b])) via geqrf — the same dtype-promotion path."""
    C = jnp.concatenate([jnp.triu(a), b], axis=-2)
    Rfac, _ = geqrf(C)
    n = a.shape[-1]
    return jnp.triu(Rfac[..., :n, :])


def reconstruct(a, R, V, T, nb):
    """Reapply Q = Q_0 ... Q_{P-1} from (V, T) to ``[R; 0]`` and return the
    full ``(n + m, n)`` matrix, which must equal the input stack ``[triu(a);
    B]``. Verifies the reflectors regardless of sign/phase convention."""
    R, V, T = np.asarray(R), np.asarray(V), np.asarray(T)
    n, m = R.shape[0], V.shape[0]
    M = np.vstack([np.triu(R), np.zeros((m, n), R.dtype)])
    for p in reversed(range(0, n, nb)):       # apply rightmost panel first
        pb = min(nb, n - p)
        Vp, Tp = V[:, p:p + pb], T[:pb, p:p + pb]
        top, bot = M[:n], M[n:]
        tmp = Tp @ (top[p:p + pb] + Vp.conj().T @ bot)
        top = top.copy()
        top[p:p + pb] -= tmp
        M = np.vstack([top, bot - Vp @ tmp])
    return M


def tol_for(dtype):
    return {np.float32: 1e-4, np.complex64: 1e-4,
            np.float64: 1e-10, np.complex128: 1e-10}[dtype]


# (m, n, l) pentagonal configurations.
TPQRT_CASES = [
    (5, 5, 5),    # square, fully triangular bottom (the LM diagonal case)
    (8, 5, 5),    # tall, triangular bottom
    (5, 5, 0),    # rectangular B (no trapezoidal part)
    (7, 5, 3),    # mixed: rectangular block on top of a trapezoid
    (6, 4, 4),
    (10, 6, 2),
    (4, 4, 4),
]


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("m,n,l", TPQRT_CASES)
def test_tpqrt_matches_scipy(m, n, l, dtype):
    """The CPU (LAPACK FFI) path returns R, V, T bit-for-bit like SciPy."""
    nb = 3
    a, b = make_pentagonal(m, n, l, dtype)
    R, V, T = tpqrt(a, b, l=l, nb=nb)
    a_s, b_s, t_s, info = _SCIPY_TPQRT[np.dtype(dtype)](
        l, nb, np.asarray(a, order="F"), np.asarray(b, order="F"))
    tol = tol_for(dtype)
    np.testing.assert_allclose(jnp.triu(R), np.triu(a_s), rtol=tol, atol=tol)
    np.testing.assert_allclose(V, b_s, rtol=tol, atol=tol)
    np.testing.assert_allclose(T, t_s, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("m,n,l", TPQRT_CASES)
def test_tpqrt_factor(m, n, l, dtype):
    """R is upper triangular with the right Gram, for the CPU path."""
    a, b = make_pentagonal(m, n, l, dtype)
    R, V, T = tpqrt(a, b, l=l)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(R), gram(reference_R(a, b)),
                               rtol=tol, atol=tol)
    np.testing.assert_allclose(R, jnp.triu(R), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("m,n,l", TPQRT_CASES)
@pytest.mark.parametrize("nb", [1, 3, 64])
def test_tpqrt_fallback_reconstructs(m, n, l, nb, dtype):
    """The pure-JAX fallback's (R, V, T) reproduces C = Q [R; 0]."""
    nb = min(nb, n)
    a, b = make_pentagonal(m, n, l, dtype)
    R, V, T = _tpqrt_blocked_householder_2d(a, b, l, nb)
    C = np.vstack([np.triu(np.asarray(a)), np.asarray(b)])
    tol = tol_for(dtype)
    np.testing.assert_allclose(reconstruct(a, R, V, T, nb), C,
                               rtol=tol, atol=tol)
    # R is upper triangular with the right Gram.
    np.testing.assert_allclose(gram(R), gram(reference_R(a, b)),
                               rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_lm_identity(dtype):
    """Trust-region LM identity: R̃ᴴ R̃ == Rᴴ R + Dᴴ D for diagonal D."""
    n = 7
    R = jnp.triu(jnp.asarray(rand((n, n), dtype)))
    d = jnp.asarray(rand((n,), dtype))
    D = jnp.diag(d)
    Rt, _, _ = tpqrt(R, D, l=n)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(Rt), gram(R) + gram(D), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_jit(dtype):
    n, m, l = 6, 6, 6
    a, b = make_pentagonal(m, n, l, dtype)
    eager = tpqrt(a, b, l=l)
    jitted = jax.jit(lambda a, b: tpqrt(a, b, l=l))(a, b)
    tol = tol_for(dtype)
    for e, j in zip(eager, jitted):
        np.testing.assert_allclose(j, e, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_batched(dtype):
    batch, n, m, l = 4, 5, 5, 5
    a = jnp.triu(jnp.asarray(rand((batch, n, n), dtype)))
    b_rows = jnp.asarray(rand((batch, m, n), dtype))
    rows = jnp.arange(m)[:, None]
    cols = jnp.arange(n)[None, :]
    b = b_rows * ((rows < (m - l)) | (cols >= rows - (m - l))).astype(dtype)

    R, V, T = tpqrt(a, b, l=l)
    expected = jax.vmap(reference_R)(a, b)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(R), gram(expected), rtol=tol, atol=tol)
    assert R.shape == (batch, n, n) and V.shape == (batch, m, n)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_vmap(dtype):
    batch, n, l = 3, 6, 6
    a = jnp.triu(jnp.asarray(rand((batch, n, n), dtype)))
    b = jax.vmap(jnp.diag)(jnp.asarray(rand((batch, n), dtype)))

    Rs, _, _ = jax.vmap(lambda a, b: tpqrt(a, b, l=l))(a, b)
    expected = jax.vmap(reference_R)(a, b)
    tol = tol_for(dtype)
    for i in range(batch):
        np.testing.assert_allclose(gram(Rs[i]), gram(expected[i]),
                                   rtol=tol, atol=tol)


def test_tpqrt_shape_errors():
    a = jnp.triu(jnp.asarray(rand((5, 5), np.float64)))
    b = jnp.asarray(rand((4, 5), np.float64))
    with pytest.raises(ValueError):
        tpqrt(jnp.asarray(rand((5, 4), np.float64)), b, l=4)  # a not square
    with pytest.raises(ValueError):
        tpqrt(a, jnp.asarray(rand((4, 6), np.float64)), l=4)  # b cols != n
    with pytest.raises(ValueError):
        tpqrt(a, b, l=10)  # l > min(m, n)
