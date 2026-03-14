API Reference
=============

``jaxtra.scipy.linalg``
-----------------------

:mod:`jaxtra.scipy.linalg` mirrors the :mod:`scipy.linalg` API, using the
primitives in :mod:`jaxtra.lax.linalg`.

.. currentmodule:: jaxtra.scipy.linalg

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   qr_multiply
   ldl
   ldl_solve

``jaxtra.lax.linalg``
---------------------

:mod:`jaxtra.lax.linalg` exposes linear algebra primitives analogous to
:mod:`jax.lax.linalg`, backed by LAPACK (CPU) and cuSPARSE (GPU).

.. currentmodule:: jaxtra.lax.linalg

.. autosummary::
   :toctree: _autosummary

   ldl
   ldl_solve
   pentadiagonal_solve
   pentadiagonal_solveh

``jaxtra._src.lax.linalg``
--------------------------

Internal wrapper functions. These are not part of the public API and may
change without notice, but are documented here for power users. Functions
in this module are not guaranteed to support automatic differentiation.

.. currentmodule:: jaxtra._src.lax.linalg

.. autosummary::
   :toctree: _autosummary

   ormqr
