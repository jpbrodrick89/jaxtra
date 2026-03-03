"""GPU cuSPARSE extension wrapper.

Mirrors ``jaxlib.gpu_sparse``'s interface: ``registrations()``,
``batch_partitionable_targets()``.
"""
from jaxtra._src.lib import _load_so

_ext = _load_so("_jaxtra_cuda", "jaxtra._jaxtra_cuda", required=False)


def registrations():
    if _ext is None or not hasattr(_ext, "sparse_registrations"):
        return {}
    return _ext.sparse_registrations()


def batch_partitionable_targets():
    if _ext is None or not hasattr(_ext, "sparse_registrations"):
        return []
    return [name for _, targets in _ext.sparse_registrations().items()
            for name, _, _ in targets if name.endswith("_ffi")]
