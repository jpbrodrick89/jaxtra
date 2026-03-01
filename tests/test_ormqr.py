"""Tests for jaxtra — ormqr primitive and qr_multiply.

Structure mirrors tests/linalg_test.py from JAX PR #35104, adapted to pytest.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import jax.scipy.linalg

jax.config.update("jax_enable_x64", True)

from jaxtra.lax.linalg import geqrf, geqp3, householder_product, ormqr
from jaxtra.scipy.linalg import qr_multiply

RNG = np.random.default_rng(42)

float_types   = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
all_dtypes    = float_types + complex_types


def rand(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)).astype(dtype)
    return RNG.standard_normal(shape).astype(dtype)


# ---------------------------------------------------------------------------
# ormqr — mirrors testOrmqr in linalg_test.py
# ---------------------------------------------------------------------------

ORMQR_CASES = [
    ((4, 3), (4, 2), True),       # Q @ C, tall A
    ((4, 4), (4, 3), True),       # Q @ C, square A
    ((6, 4), (6, 5), True),       # Q @ C, tall A
    ((4, 3), (2, 4), False),      # C @ Q, tall A
    ((4, 4), (3, 4), False),      # C @ Q, square A
    ((3, 6, 4), (3, 6, 5), True), # batched Q @ C
]


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("a_shape,c_shape,left", ORMQR_CASES)
def test_ormqr(a_shape, c_shape, dtype, left, transpose):
    a = jnp.array(rand(a_shape, dtype))
    c = jnp.array(rand(c_shape, dtype))

    qr_result, taus = geqrf(a)
    result = ormqr(qr_result, taus, c, left=left, transpose=transpose)

    # Reference: build full Q from householder_product, then matmul.
    m, n = a_shape[-2:]
    if m > n:
        padded = jnp.pad(qr_result,
                         [(0, 0)] * (qr_result.ndim - 1) + [(0, m - n)])
        q = householder_product(padded, taus)
    elif m < n:
        q = householder_product(qr_result[..., :m, :m], taus)
    else:
        q = householder_product(qr_result, taus)
    q_op = jnp.conj(jnp.swapaxes(q, -1, -2)) if transpose else q
    expected = q_op @ c if left else c @ q_op

    tol = {np.float32: 1e-4, np.complex64: 1e-4,
           np.float64: 1e-10, np.complex128: 1e-10}[dtype]
    np.testing.assert_allclose(result, expected, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# qr_multiply — mirrors testQrMultiply in linalg_test.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("mode", ["right", "left"])
@pytest.mark.parametrize("pivoting", [False, True])
@pytest.mark.parametrize("shape", [(4, 3), (4, 4), (6, 4), (3, 6, 4)])
def test_qr_multiply(shape, dtype, mode, pivoting):
    m, n = shape[-2:]
    a = jnp.array(rand(shape, dtype))
    k = min(m, n)
    c_shape = shape[:-2] + (k, 2) if mode == "left" else shape[:-2] + (2, m)
    c = jnp.array(rand(c_shape, dtype))

    if pivoting:
        result, r, p = qr_multiply(a, c, mode=mode, pivoting=True)
    else:
        result, r = qr_multiply(a, c, mode=mode, pivoting=False)

    # Reference via jax.scipy.linalg.qr
    if pivoting:
        q_full, r_full, _ = jax.scipy.linalg.qr(a, pivoting=True)
    else:
        q_full, r_full = jax.scipy.linalg.qr(a)
    r_ref = r_full[..., :k, :]

    expected = q_full[..., :k] @ c if mode == "left" else c @ q_full[..., :k]

    tol = {np.float32: 1e-4, np.complex64: 1e-4,
           np.float64: 1e-10, np.complex128: 1e-10}[dtype]
    np.testing.assert_allclose(result, expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(r, r_ref, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# qr_multiply 1-D c — mirrors testQrMultiply1D in linalg_test.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("mode", ["right", "left"])
@pytest.mark.parametrize("shape", [(4, 3), (4, 4), (6, 4)])
def test_qr_multiply_1d(shape, dtype, mode):
    m, n = shape[-2:]
    k = min(m, n)
    a = jnp.array(rand(shape, dtype))
    c = jnp.array(rand((k,), dtype) if mode == "left" else rand((m,), dtype))

    result, r = qr_multiply(a, c, mode=mode)
    assert result.ndim == 1

    q_full, _ = jax.scipy.linalg.qr(a)
    if mode == "left":
        expected = (q_full[..., :k] @ c[:, None]).ravel()
    else:
        expected = (c[None, :] @ q_full[..., :k]).ravel()

    tol = {np.float32: 1e-4, np.complex64: 1e-4,
           np.float64: 1e-10, np.complex128: 1e-10}[dtype]
    np.testing.assert_allclose(result, expected, rtol=tol, atol=tol)


# ---------------------------------------------------------------------------
# Extra: JIT and vmap (not in PR — jtu._CompileAndCheck covers this there)
# ---------------------------------------------------------------------------

def test_ormqr_jit():
    a = jnp.array(rand((6, 3), np.float64))
    c = jnp.array(rand((6, 4), np.float64))
    H, taus = geqrf(a)
    ref = ormqr(H, taus, c, left=True, transpose=True)

    @jax.jit
    def fn(H, taus, c):
        return ormqr(H, taus, c, left=True, transpose=True)

    result = fn(H, taus, c)
    np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)


def test_ormqr_vmap():
    batch, m, k = 4, 5, 3
    a_batch = jnp.array(rand((batch, m, k), np.float64))
    c_batch = jnp.array(rand((batch, m, m), np.float64))

    @jax.vmap
    def fn(a, c):
        H, taus = geqrf(a)
        return ormqr(H, taus, c, left=True, transpose=False)

    result = fn(a_batch, c_batch)
    assert result.shape == (batch, m, m)

    # Validate each slice independently
    for i in range(batch):
        H, taus = geqrf(a_batch[i])
        ref_i = ormqr(H, taus, c_batch[i], left=True, transpose=False)
        np.testing.assert_allclose(result[i], ref_i, rtol=1e-10, atol=1e-10)


def test_qr_multiply_jit():
    a = jnp.array(rand((6, 4), np.float64))
    c = jnp.array(rand((2, 6), np.float64))
    ref, R = qr_multiply(a, c, mode="right")

    @jax.jit
    def fn(a, c):
        return qr_multiply(a, c, mode="right")

    result, R2 = fn(a, c)
    np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(R2, R, rtol=1e-10, atol=1e-10)
