"""``jaxtra.lax.linalg`` — linear algebra primitives.

Currently exposes:

* :func:`pentadiagonal_solve` — pentadiagonal banded linear solve backed by
  LAPACK ``gbsv`` (CPU) and cuSPARSE ``gpsvInterleavedBatch`` (GPU).
* :func:`pentadiagonal_solveh` — Hermitian/SPD pentadiagonal solve backed by
  LAPACK ``pbsv`` (CPU, banded Cholesky) and cuSPARSE
  ``gpsvInterleavedBatch`` (GPU).
"""
from __future__ import annotations

from jaxtra._src.lax.linalg import pentadiagonal_solve, pentadiagonal_solveh

__all__ = ["pentadiagonal_solve", "pentadiagonal_solveh"]
