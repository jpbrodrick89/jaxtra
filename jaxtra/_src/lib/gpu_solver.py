"""GPU cuSolver extension wrapper.

Mirrors ``jaxlib.gpu_solver``'s interface: ``registrations()``,
``batch_partitionable_targets()``.
"""
from jaxtra._src.lib import _load_so

_ext = _load_so("_jaxtra_cuda", "jaxtra._jaxtra_cuda", required=False)


def registrations():
    return _ext.registrations() if _ext is not None else {}


def batch_partitionable_targets():
    if _ext is None:
        return []
    return [name for _, targets in _ext.registrations().items()
            for name, _, _ in targets if name.endswith("_ffi")]
