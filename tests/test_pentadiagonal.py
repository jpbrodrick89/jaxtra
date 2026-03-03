"""Tests for jaxtra — pentadiagonal_solve primitive.

Covers:
  - Correctness vs numpy LU for random 10×10 pentadiagonal systems.
  - Correctness vs numpy LU for a hyperdiffusion (biharmonic) operator.
  - JIT compilation.
  - Batched execution via vmap.
  - Forward-mode AD (JVP).
  - Reverse-mode AD (grad).
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxtra._src.lax.linalg import pentadiagonal_solve

RNG = np.random.default_rng(17)

float_types   = [np.float32, np.float64]
complex_types = [np.complex64, np.complex128]
all_dtypes    = float_types + complex_types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand(shape, dtype):
    if np.issubdtype(dtype, np.complexfloating):
        return (RNG.standard_normal(shape) + 1j * RNG.standard_normal(shape)).astype(dtype)
    return RNG.standard_normal(shape).astype(dtype)


def make_diag_dominant_penta(n, dtype):
    """Random diagonally dominant pentadiagonal system (guaranteed solvable)."""
    ds = rand(n, dtype) * 0.1
    dl = rand(n, dtype) * 0.2
    # Main diagonal: large positive real part so the system is nonsingular.
    if np.issubdtype(dtype, np.complexfloating):
        d = rand(n, dtype) * 0.5 + (10.0 + 0j)
    else:
        d = rand(n, dtype) * 0.5 + 10.0
    du = rand(n, dtype) * 0.2
    dw = rand(n, dtype) * 0.1
    b  = rand(n, dtype)
    return ds, dl, d, du, dw, b


def make_hyperdiffusion(n, dtype):
    """SPD biharmonic stencil: [1, -4, 8, -4, 1] (diagonally dominant for safety)."""
    # Biharmonic fourth-difference operator with a slightly larger main diagonal
    # to ensure strict diagonal dominance regardless of boundary effects.
    ds = np.ones(n, dtype=dtype)
    dl = np.full(n, -4.0, dtype=dtype)
    d  = np.full(n,  8.0, dtype=dtype)
    du = np.full(n, -4.0, dtype=dtype)
    dw = np.ones(n, dtype=dtype)
    b  = rand(n, dtype)
    return ds, dl, d, du, dw, b


def diags_to_dense(ds, dl, d, du, dw):
    """Reconstruct dense n×n matrix from five diagonals (numpy)."""
    n = len(d)
    A = np.diag(d)
    for i in range(1, n):
        A[i, i - 1] += dl[i]
    for i in range(2, n):
        A[i, i - 2] += ds[i]
    for i in range(n - 1):
        A[i, i + 1] += du[i]
    for i in range(n - 2):
        A[i, i + 2] += dw[i]
    return A


def lu_reference(ds, dl, d, du, dw, b):
    """Dense numpy LU solve as reference."""
    A = diags_to_dense(
        np.array(ds), np.array(dl), np.array(d), np.array(du), np.array(dw))
    return np.linalg.solve(A, np.array(b))


def tol(dtype):
    return {np.float32: 1e-4, np.complex64: 1e-4,
            np.float64: 1e-10, np.complex128: 1e-10}[dtype]


# ---------------------------------------------------------------------------
# Correctness — random diagonally dominant pentadiagonal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("n", [5, 10, 50])
def test_random_pentadiagonal(n, dtype):
    """pentadiagonal_solve matches numpy LU for a random system."""
    ds, dl, d, du, dw, b = make_diag_dominant_penta(n, dtype)
    x     = pentadiagonal_solve(
        jnp.array(ds), jnp.array(dl), jnp.array(d),
        jnp.array(du), jnp.array(dw), jnp.array(b))
    x_ref = lu_reference(ds, dl, d, du, dw, b)
    atol = rtol = tol(dtype)
    np.testing.assert_allclose(np.array(x), x_ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Correctness — hyperdiffusion operator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", all_dtypes)
def test_hyperdiffusion(dtype):
    """pentadiagonal_solve matches numpy LU for the biharmonic stencil."""
    n = 10
    ds, dl, d, du, dw, b = make_hyperdiffusion(n, dtype)
    x     = pentadiagonal_solve(
        jnp.array(ds), jnp.array(dl), jnp.array(d),
        jnp.array(du), jnp.array(dw), jnp.array(b))
    x_ref = lu_reference(ds, dl, d, du, dw, b)
    atol = rtol = tol(dtype)
    np.testing.assert_allclose(np.array(x), x_ref, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------

def test_jit():
    """JIT-compiled pentadiagonal_solve matches eager result."""
    n = 10
    ds, dl, d, du, dw, b = make_diag_dominant_penta(n, np.float64)
    args = tuple(jnp.array(v) for v in (ds, dl, d, du, dw, b))

    ref    = pentadiagonal_solve(*args)
    result = jax.jit(pentadiagonal_solve)(*args)
    np.testing.assert_allclose(np.array(result), np.array(ref), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------

def test_vmap():
    """Batched pentadiagonal_solve via vmap matches per-sample eager results."""
    batch, n = 4, 10
    ds_b = jnp.array(rand((batch, n), np.float64))
    dl_b = jnp.array(rand((batch, n), np.float64))
    d_b  = jnp.array(rand((batch, n), np.float64) * 0.5 + 10.0)
    du_b = jnp.array(rand((batch, n), np.float64))
    dw_b = jnp.array(rand((batch, n), np.float64))
    b_b  = jnp.array(rand((batch, n), np.float64))

    result = jax.vmap(pentadiagonal_solve)(ds_b, dl_b, d_b, du_b, dw_b, b_b)
    assert result.shape == (batch, n)

    for i in range(batch):
        ref_i = pentadiagonal_solve(ds_b[i], dl_b[i], d_b[i],
                                    du_b[i], dw_b[i], b_b[i])
        np.testing.assert_allclose(
            np.array(result[i]), np.array(ref_i), rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# Forward-mode AD (JVP)
# ---------------------------------------------------------------------------

def test_jvp_wrt_b():
    """JVP with respect to b matches finite-difference estimate."""
    n = 6
    ds, dl, d, du, dw, b = make_diag_dominant_penta(n, np.float64)
    ds_j, dl_j, d_j, du_j, dw_j, b_j = (jnp.array(v)
                                          for v in (ds, dl, d, du, dw, b))
    db = jnp.array(rand(n, np.float64))

    def solve_b(b_):
        return pentadiagonal_solve(ds_j, dl_j, d_j, du_j, dw_j, b_)

    _, tangent = jax.jvp(solve_b, (b_j,), (db,))

    # Finite-difference reference
    eps = 1e-6
    fd = (solve_b(b_j + eps * db) - solve_b(b_j - eps * db)) / (2.0 * eps)
    np.testing.assert_allclose(np.array(tangent), np.array(fd), rtol=1e-5, atol=1e-5)


def test_jvp_wrt_d():
    """JVP with respect to the main diagonal d matches finite-difference estimate."""
    n = 6
    ds, dl, d, du, dw, b = make_diag_dominant_penta(n, np.float64)
    ds_j, dl_j, d_j, du_j, dw_j, b_j = (jnp.array(v)
                                          for v in (ds, dl, d, du, dw, b))
    dd = jnp.array(rand(n, np.float64) * 0.01)  # small perturbation

    def solve_d(d_):
        return pentadiagonal_solve(ds_j, dl_j, d_, du_j, dw_j, b_j)

    _, tangent = jax.jvp(solve_d, (d_j,), (dd,))

    eps = 1e-6
    fd = (solve_d(d_j + eps * dd) - solve_d(d_j - eps * dd)) / (2.0 * eps)
    np.testing.assert_allclose(np.array(tangent), np.array(fd), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Reverse-mode AD (grad)
# ---------------------------------------------------------------------------

def test_grad_wrt_b():
    """jax.grad with respect to b matches finite-difference gradient."""
    n = 6
    ds, dl, d, du, dw, b = make_diag_dominant_penta(n, np.float64)
    ds_j, dl_j, d_j, du_j, dw_j, b_j = (jnp.array(v)
                                          for v in (ds, dl, d, du, dw, b))

    def loss(b_):
        x = pentadiagonal_solve(ds_j, dl_j, d_j, du_j, dw_j, b_)
        return jnp.sum(x ** 2)

    grad_b = jax.grad(loss)(b_j)

    # Finite-difference reference
    eps = 1e-6
    fd_grad = np.zeros(n)
    for k in range(n):
        e = np.zeros(n, dtype=np.float64)
        e[k] = eps
        fd_grad[k] = (loss(b_j + e) - loss(b_j - e)) / (2.0 * eps)

    np.testing.assert_allclose(np.array(grad_b), fd_grad, rtol=1e-5, atol=1e-5)


def test_grad_wrt_diagonals():
    """jax.grad with respect to all five diagonals matches finite differences."""
    n = 6
    ds, dl, d, du, dw, b = make_diag_dominant_penta(n, np.float64)
    ds_j, dl_j, d_j, du_j, dw_j, b_j = (jnp.array(v)
                                          for v in (ds, dl, d, du, dw, b))

    def loss(ds_, dl_, d_, du_, dw_):
        x = pentadiagonal_solve(ds_, dl_, d_, du_, dw_, b_j)
        return jnp.sum(x ** 2)

    grads = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        ds_j, dl_j, d_j, du_j, dw_j)

    eps = 1e-6
    diag_args = [ds_j, dl_j, d_j, du_j, dw_j]
    for arg_idx, g in enumerate(grads):
        fd_g = np.zeros(n)
        for k in range(n):
            e = np.zeros(n, dtype=np.float64)
            e[k] = eps
            args_p = diag_args.copy(); args_p[arg_idx] = diag_args[arg_idx] + e
            args_m = diag_args.copy(); args_m[arg_idx] = diag_args[arg_idx] - e
            fd_g[k] = (loss(*args_p) - loss(*args_m)) / (2.0 * eps)
        np.testing.assert_allclose(
            np.array(g), fd_g, rtol=1e-4, atol=1e-4,
            err_msg=f"gradient mismatch for diagonal arg {arg_idx}")
