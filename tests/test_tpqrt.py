"""Tests for jaxtra — tpqrt (triangular-pentagonal QR) primitive.

Covers the LAPACK CPU path and the pure-JAX block-Givens fallback against a
``geqrf``-based reference, across dtypes, pentagonal shapes, jit and vmap.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jax._src.lax.linalg import geqrf
from jaxtra._src.lax.linalg import tpqrt, _tpqrt_givens_2d

RNG = np.random.default_rng(42)

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
all_dtypes = float_types + complex_types


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
def test_tpqrt_lapack(m, n, l, dtype):
    a, b = make_pentagonal(m, n, l, dtype)
    R = tpqrt(a, b, l=l)
    expected = reference_R(a, b)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(R), gram(expected), rtol=tol, atol=tol)
    # R must be upper triangular.
    np.testing.assert_allclose(R, jnp.triu(R), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("m,n,l", TPQRT_CASES)
def test_tpqrt_givens_fallback(m, n, l, dtype):
    a, b = make_pentagonal(m, n, l, dtype)
    R = _tpqrt_givens_2d(a, b, l)
    expected = reference_R(a, b)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(R), gram(expected), rtol=tol, atol=tol)
    np.testing.assert_allclose(R, jnp.triu(R), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_lm_identity(dtype):
    """Trust-region LM identity: R̃ᴴ R̃ == Rᴴ R + Dᴴ D for diagonal D."""
    n = 7
    R = jnp.triu(jnp.asarray(rand((n, n), dtype)))
    d = jnp.asarray(rand((n,), dtype))
    D = jnp.diag(d)
    Rt = tpqrt(R, D, l=n)
    expected = gram(R) + gram(D)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(Rt), expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_jit(dtype):
    n, m, l = 6, 6, 6
    a, b = make_pentagonal(m, n, l, dtype)
    eager = tpqrt(a, b, l=l)
    jitted = jax.jit(lambda a, b: tpqrt(a, b, l=l))(a, b)
    tol = tol_for(dtype)
    np.testing.assert_allclose(jitted, eager, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_batched(dtype):
    batch, n, m, l = 4, 5, 5, 5
    a = jnp.triu(jnp.asarray(rand((batch, n, n), dtype)))
    b_rows = jnp.asarray(rand((batch, m, n), dtype))
    rows = jnp.arange(m)[:, None]
    cols = jnp.arange(n)[None, :]
    b = b_rows * ((rows < (m - l)) | (cols >= rows - (m - l))).astype(dtype)

    R = tpqrt(a, b, l=l)
    expected = jax.vmap(reference_R)(a, b)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(R), gram(expected), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_vmap(dtype):
    batch, n, m, l = 3, 6, 6, 6
    a = jnp.triu(jnp.asarray(rand((batch, n, n), dtype)))
    b = jax.vmap(jnp.diag)(jnp.asarray(rand((batch, n), dtype)))

    out = jax.vmap(lambda a, b: tpqrt(a, b, l=l))(a, b)
    expected = jax.vmap(reference_R)(a, b)
    tol = tol_for(dtype)
    for i in range(batch):
        np.testing.assert_allclose(gram(out[i]), gram(expected[i]),
                                   rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_tpqrt_lapack_matches_givens(dtype):
    """The CPU LAPACK path and the pure-JAX fallback agree (Gram-invariant)."""
    m, n, l = 9, 6, 4
    a, b = make_pentagonal(m, n, l, dtype)
    R_lapack = tpqrt(a, b, l=l)
    R_givens = _tpqrt_givens_2d(a, b, l)
    tol = tol_for(dtype)
    np.testing.assert_allclose(gram(R_lapack), gram(R_givens),
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
