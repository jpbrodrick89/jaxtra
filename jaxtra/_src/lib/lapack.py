"""CPU LAPACK extension wrapper.

Mirrors ``jaxlib.lapack``'s interface: ``registrations()``,
``batch_partitionable_targets()``, ``prepare_lapack_call()``.
"""

from jaxlib import lapack as _jaxlib_lapack

from jaxtra._src.lib import _load_so

_ext = _load_so("_jaxtra", "jaxtra._jaxtra", required=True)
_ext.initialize()


def registrations():
  return _ext.registrations()


def batch_partitionable_targets():
  return [
    name
    for _, targets in _ext.registrations().items()
    for name, _, _ in targets
    if name.endswith("_ffi")
  ]


# Same naming convention as jaxlib — no need to redefine.
prepare_lapack_call = _jaxlib_lapack.prepare_lapack_call
