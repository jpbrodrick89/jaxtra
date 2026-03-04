"""jaxtra: JAX extensions for LAPACK routines and GPU-accelerated linear algebra.

Registers XLA FFI kernels as proper JAX primitives — JIT, vmap, and grad
compatible — backed by LAPACK (CPU) and cuSOLVER/cuSPARSE (GPU).

Currently exposed:
  • ``qr_multiply`` — combined QR decomposition + Q-multiply, backed by LAPACK
                      dormqr/sormqr/cunmqr/zunmqr, available at
                      ``jaxtra.scipy.linalg.qr_multiply``.
  • ``pentadiagonal_solve`` — pentadiagonal banded linear solve A x = b, backed
                      by LAPACK ``gbsv`` (CPU) and cuSPARSE
                      ``gpsvInterleavedBatch`` (GPU), available at
                      ``jaxtra.lax.linalg.pentadiagonal_solve``.
  • ``pentadiagonal_solveh`` — Hermitian/SPD pentadiagonal solve A x = b, backed
                      by LAPACK ``pbsv`` (CPU, banded Cholesky) and cuSPARSE
                      ``gpsvInterleavedBatch`` (GPU), available at
                      ``jaxtra.lax.linalg.pentadiagonal_solveh``.

Quick start
-----------
>>> import jax.numpy as jnp
>>> import jaxtra.scipy.linalg as jsla
>>>
>>> A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
>>> b = jnp.array([2., 4., 5., 4.])
>>> Qtb, R = jsla.qr_multiply(A, b, mode='right')
"""
