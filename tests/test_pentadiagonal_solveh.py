"""Tests for jaxtra — pentadiagonal_solveh primitive (Hermitian banded solve).

Covers:
  - Correctness vs numpy dense solve for SPD systems (all dtypes).
  - Correctness for Hermitian (complex) systems.
  - Multi-RHS support (rank-2 b).
  - JIT compilation.
  - Batched execution via vmap (all args batched + only-b batched).
  - Forward-mode AD (JVP).
  - Reverse-mode AD (grad).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import pentadiagonal_solveh

RNG = np.random.default_rng(42)

float_types = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
all_dtypes = float_types + complex_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return (RNG.standard_normal(shape)
                + 1j * RNG.standard_normal(shape)).astype(dtype)
    return RNG.standard_normal(shape).astype(dtype)


def make_spd_penta(n, dtype, nrhs=1):
    """Build an SPD pentadiagonal system using the biharmonic stencil.

    Returns (d, du, dw, b) — upper-triangle diagonals only.
    """
    eps = 0.1  # shift for strict positive definiteness
    d = np.full(n, 8.0 + eps, dtype=dtype)
    du = np.full(n, -4.0, dtype=dtype)
    dw = np.ones(n, dtype=dtype)
    b = rand((n, nrhs), dtype)
    return d, du, dw, b


def make_hermitian_penta(n, dtype, nrhs=1):
    """Build a Hermitian (complex) positive definite pentadiagonal system.

    Upper diagonals are complex; diagonal is real (as required for Hermitian).
    """
    # Real diagonal, large enough for positive definiteness.
    d_real = np.full(n, 10.0)
    d = d_real.astype(dtype)
    du = (rand(n, dtype) * 0.2)
    dw = (rand(n, dtype) * 0.1)
    b = rand((n, nrhs), dtype)
    return d, du, dw, b


def upper_diags_to_dense(d, du, dw):
    """Reconstruct dense Hermitian n×n matrix from upper diagonals."""
    n = len(d)
    A = np.diag(d.astype(np.complex128 if np.iscomplexobj(d) else np.float64))
    for i in range(n - 1):
        A[i, i + 1] += du[i]
        A[i + 1, i] += np.conj(du[i])
    for i in range(n - 2):
        A[i, i + 2] += dw[i]
        A[i + 2, i] += np.conj(dw[i])
    return A.astype(d.dtype)


def dense_reference(d, du, dw, b):
    """Dense numpy solve as reference."""
    A = upper_diags_to_dense(np.array(d), np.array(du), np.array(dw))
    return np.linalg.solve(A, np.array(b))


def tol(dtype):
    return {np.float32: 1e-4, np.complex64: 1e-4,
            np.float64: 1e-10, np.complex128: 1e-10}[dtype]


# ---------------------------------------------------------------------------
# Correctness — SPD biharmonic stencil
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("n", [5, 10, 50])
def test_spd_pentadiagonal(n, dtype):
    """pentadiagonal_solveh matches numpy for an SPD system."""
    d, du, dw, b = make_spd_penta(n, dtype)
    x = pentadiagonal_solveh(
        jnp.array(d), jnp.array(du), jnp.array(dw), jnp.array(b))
    x_ref = dense_reference(d, du, dw, b)
    np.testing.assert_allclose(
        np.array(x), x_ref, rtol=tol(dtype), atol=tol(dtype))


# ---------------------------------------------------------------------------
# Correctness — Hermitian (complex) system
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", complex_types)
def test_hermitian_complex(dtype):
    """pentadiagonal_solveh matches numpy for a complex Hermitian system."""
    n = 10
    d, du, dw, b = make_hermitian_penta(n, dtype)
    x = pentadiagonal_solveh(
        jnp.array(d), jnp.array(du), jnp.array(dw), jnp.array(b))
    x_ref = dense_reference(d, du, dw, b)
    np.testing.assert_allclose(
        np.array(x), x_ref, rtol=tol(dtype), atol=tol(dtype))


# ---------------------------------------------------------------------------
# Multi-RHS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("nrhs", [1, 3, 7])
def test_multi_rhs(nrhs, dtype):
    """pentadiagonal_solveh matches numpy for multiple RHS columns."""
    n = 10
    d, du, dw, b = make_spd_penta(n, dtype, nrhs=nrhs)
    x = pentadiagonal_solveh(
        jnp.array(d), jnp.array(du), jnp.array(dw), jnp.array(b))
    x_ref = dense_reference(d, du, dw, b)
    np.testing.assert_allclose(
        np.array(x), x_ref, rtol=tol(dtype), atol=tol(dtype))


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------

def test_jit():
    """JIT-compiled pentadiagonal_solveh matches eager result."""
    n = 10
    d, du, dw, b = make_spd_penta(n, np.float64)
    args = tuple(jnp.array(v) for v in (d, du, dw, b))

    ref = pentadiagonal_solveh(*args)
    result = jax.jit(pentadiagonal_solveh)(*args)
    np.testing.assert_allclose(
        np.array(result), np.array(ref), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

def test_vmap():
    """Batched pentadiagonal_solveh via vmap matches per-sample results."""
    batch, n, nrhs = 4, 10, 1
    d_b = jnp.array(np.full((batch, n), 10.0))
    du_b = jnp.array(rand((batch, n), np.float64) * 0.2)
    dw_b = jnp.array(rand((batch, n), np.float64) * 0.1)
    b_b = jnp.array(rand((batch, n, nrhs), np.float64))

    result = jax.vmap(pentadiagonal_solveh)(d_b, du_b, dw_b, b_b)
    assert result.shape == (batch, n, nrhs)

    for i in range(batch):
        ref_i = pentadiagonal_solveh(d_b[i], du_b[i], dw_b[i], b_b[i])
        np.testing.assert_allclose(
            np.array(result[i]), np.array(ref_i), rtol=1e-10, atol=1e-10)


def test_vmap_over_b():
    """vmap with only b batched uses the efficient nrhs-folding path."""
    n, nrhs = 10, 5
    d, du, dw, _ = make_spd_penta(n, np.float64)
    d_j, du_j, dw_j = (jnp.array(v) for v in (d, du, dw))
    b_batch = jnp.array(rand((nrhs, n, 1), np.float64))

    result = jax.vmap(
        pentadiagonal_solveh,
        in_axes=(None, None, None, 0))(d_j, du_j, dw_j, b_batch)
    assert result.shape == (nrhs, n, 1)

    for i in range(nrhs):
        ref_i = pentadiagonal_solveh(d_j, du_j, dw_j, b_batch[i])
        np.testing.assert_allclose(
            np.array(result[i]), np.array(ref_i), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Forward-mode AD (JVP)
# ---------------------------------------------------------------------------

def test_jvp_wrt_b():
    """JVP with respect to b matches finite-difference estimate."""
    n = 6
    d, du, dw, b = make_spd_penta(n, np.float64)
    d_j, du_j, dw_j, b_j = (jnp.array(v) for v in (d, du, dw, b))
    db = jnp.array(rand((n, 1), np.float64))

    def solve_b(b_):
        return pentadiagonal_solveh(d_j, du_j, dw_j, b_)

    _, tangent = jax.jvp(solve_b, (b_j,), (db,))

    eps = 1e-6
    fd = (solve_b(b_j + eps * db) - solve_b(b_j - eps * db)) / (2.0 * eps)
    np.testing.assert_allclose(
        np.array(tangent), np.array(fd), rtol=1e-5, atol=1e-5)


def test_jvp_wrt_d():
    """JVP with respect to the main diagonal matches finite-difference."""
    n = 6
    d, du, dw, b = make_spd_penta(n, np.float64)
    d_j, du_j, dw_j, b_j = (jnp.array(v) for v in (d, du, dw, b))
    dd = jnp.array(rand(n, np.float64) * 0.01)

    def solve_d(d_):
        return pentadiagonal_solveh(d_, du_j, dw_j, b_j)

    _, tangent = jax.jvp(solve_d, (d_j,), (dd,))

    eps = 1e-6
    fd = (solve_d(d_j + eps * dd) - solve_d(d_j - eps * dd)) / (2.0 * eps)
    np.testing.assert_allclose(
        np.array(tangent), np.array(fd), rtol=1e-5, atol=1e-5)


def test_jvp_multi_rhs():
    """JVP with multi-RHS b matches finite-difference estimate."""
    n, nrhs = 6, 3
    d, du, dw, b = make_spd_penta(n, np.float64, nrhs=nrhs)
    d_j, du_j, dw_j, b_j = (jnp.array(v) for v in (d, du, dw, b))
    db = jnp.array(rand((n, nrhs), np.float64))

    def solve_b(b_):
        return pentadiagonal_solveh(d_j, du_j, dw_j, b_)

    _, tangent = jax.jvp(solve_b, (b_j,), (db,))

    eps = 1e-6
    fd = (solve_b(b_j + eps * db) - solve_b(b_j - eps * db)) / (2.0 * eps)
    np.testing.assert_allclose(
        np.array(tangent), np.array(fd), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Reverse-mode AD (grad)
# ---------------------------------------------------------------------------

def test_grad_wrt_b():
    """jax.grad with respect to b matches finite-difference gradient."""
    n = 6
    d, du, dw, b = make_spd_penta(n, np.float64)
    d_j, du_j, dw_j, b_j = (jnp.array(v) for v in (d, du, dw, b))

    def loss(b_):
        x = pentadiagonal_solveh(d_j, du_j, dw_j, b_)
        return jnp.sum(x ** 2)

    grad_b = jax.grad(loss)(b_j)

    eps = 1e-6
    fd_grad = np.zeros((n, 1))
    for k in range(n):
        e = np.zeros((n, 1), dtype=np.float64)
        e[k, 0] = eps
        fd_grad[k, 0] = (loss(b_j + e) - loss(b_j - e)) / (2.0 * eps)

    np.testing.assert_allclose(
        np.array(grad_b), fd_grad, rtol=1e-5, atol=1e-5)


def test_grad_wrt_diagonals():
    """jax.grad with respect to all three diagonals matches finite diffs."""
    n = 6
    d, du, dw, b = make_spd_penta(n, np.float64)
    d_j, du_j, dw_j, b_j = (jnp.array(v) for v in (d, du, dw, b))

    def loss(d_, du_, dw_):
        x = pentadiagonal_solveh(d_, du_, dw_, b_j)
        return jnp.sum(x ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2))(d_j, du_j, dw_j)

    eps = 1e-6
    diag_args = [d_j, du_j, dw_j]
    for arg_idx, g in enumerate(grads):
        fd_g = np.zeros(n)
        for k in range(n):
            e = np.zeros(n, dtype=np.float64)
            e[k] = eps
            args_p = diag_args.copy()
            args_p[arg_idx] = diag_args[arg_idx] + e
            args_m = diag_args.copy()
            args_m[arg_idx] = diag_args[arg_idx] - e
            fd_g[k] = (loss(*args_p) - loss(*args_m)) / (2.0 * eps)
        np.testing.assert_allclose(
            np.array(g), fd_g, rtol=1e-4, atol=1e-4,
            err_msg=f"gradient mismatch for diagonal arg {arg_idx}")


def test_grad_multi_rhs():
    """jax.grad with multi-RHS b matches finite-difference gradient."""
    n, nrhs = 6, 3
    d, du, dw, b = make_spd_penta(n, np.float64, nrhs=nrhs)
    d_j, du_j, dw_j, b_j = (jnp.array(v) for v in (d, du, dw, b))

    def loss(b_):
        x = pentadiagonal_solveh(d_j, du_j, dw_j, b_)
        return jnp.sum(x ** 2)

    grad_b = jax.grad(loss)(b_j)

    eps = 1e-6
    fd_grad = np.zeros((n, nrhs))
    for k in range(n):
        for j in range(nrhs):
            e = np.zeros((n, nrhs), dtype=np.float64)
            e[k, j] = eps
            fd_grad[k, j] = (loss(b_j + e) - loss(b_j - e)) / (2.0 * eps)

    np.testing.assert_allclose(
        np.array(grad_b), fd_grad, rtol=1e-5, atol=1e-5)
