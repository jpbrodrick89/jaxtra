API Reference
=============

.. currentmodule:: jaxtra

``jaxtra``
----------

Top-level convenience imports — the most common entry point.

.. autosummary::

   ormqr
   ormqr_lapack

.. autofunction:: ormqr

.. autofunction:: ormqr_lapack

----

``jaxtra.scipy.linalg``
-----------------------

Mirrors the interface of :mod:`jax.scipy.linalg`.

.. currentmodule:: jaxtra.scipy.linalg

.. autosummary::

   qr_multiply

.. autofunction:: qr_multiply

----

``jaxtra.lax.linalg``
---------------------

Mirrors the interface of :mod:`jax.lax.linalg`.

.. currentmodule:: jaxtra.lax.linalg

.. autosummary::

   ormqr
   geqrf
   geqp3
   householder_product

.. autofunction:: ormqr

The following are re-exported directly from JAX — see the
`JAX API docs <https://jax.readthedocs.io/en/latest/lax.linalg.html>`_
for full documentation.

.. autosummary::

   geqrf
   geqp3
   householder_product
