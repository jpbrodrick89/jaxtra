"""GPU hybrid extension wrapper.

Provides LAPACK hetrf/hetrs running on CPU with explicit GPU↔CPU memory
transfers.  Mirrors ``jaxtra._src.lib.gpu_solver``'s interface.
"""

from jaxtra._src.lib import _load_so

_ext = _load_so("_jaxtra_hybrid", "jaxtra._jaxtra_hybrid", required=False)
if _ext is not None:
  _ext.initialize()


def registrations():
  return _ext.registrations() if _ext is not None else {}
