"""
jaxtra._core — JAX primitive for LAPACK ORMQR via the XLA FFI.

Registers the four dtype variants (f32/f64/c64/c128) with JAX's FFI and
exposes ``ormqr``, a proper JAX primitive that:

* participates in ``jax.jit`` and ``jax.vmap``
* applies Q from a compact QR factorisation (Householder vectors + taus)
  to a matrix C **without ever forming Q**
* works on CPU today; GPU support requires a cuSOLVER kernel (future work)

For non-JAX (pure numpy) use, see ``jaxtra._numpy_lapack.ormqr_lapack``.
"""
from __future__ import annotations

import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax._src import core     # jax.core.Primitive was removed in 0.9
from jax._src import dispatch
from jax._src.interpreters import mlir, batching
import jax.ffi as ffi

# ---------------------------------------------------------------------------
# 1. Register FFI targets from the C extension
# ---------------------------------------------------------------------------

def _load_extension():
    """Load _jaxtra.so by file path, bypassing the package import system.

    This avoids the circular import that would occur if we used
    ``from jaxtra import _jaxtra`` while ``jaxtra/__init__.py`` is still
    being initialised.

    Search order:
      1. ``jaxtra/`` (installed wheel / cmake --install)
      2. Project root one level up (scikit-build-core inplace / editable mode)
    """
    import glob
    import importlib.util
    import sys

    mod_name = "jaxtra._jaxtra"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    here = os.path.dirname(__file__)
    search_dirs = [here, os.path.dirname(here)]
    candidates = []
    for d in search_dirs:
        candidates.extend(glob.glob(os.path.join(d, "_jaxtra*.so")))

    if not candidates:
        raise ImportError(
            "jaxtra C extension (_jaxtra.so) not found. "
            "Build it with:  pip install -e . --no-build-isolation"
        )
    spec = importlib.util.spec_from_file_location(mod_name, candidates[0])
    ext = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = ext
    spec.loader.exec_module(ext)
    return ext


def _register_targets() -> None:
    """Register XLA FFI targets for all dtype variants.

    Mirrors jaxlib's initialize() mechanism: import scipy to force its bundled
    OpenBLAS into the process, then extract raw LAPACK function pointers from
    scipy.linalg.cython_lapack.__pyx_capi__ (PyCapsules) via ctypes and hand
    them to the C++ extension.  No LAPACK link at wheel-build time.
    """
    import ctypes

    _jaxtra = _load_extension()

    # Force scipy's OpenBLAS into the process and get its LAPACK capsules.
    import scipy.linalg.cython_lapack as _cl  # noqa: triggers libopenblas load
    _capi = _cl.__pyx_capi__

    def _ptr(name: str) -> int:
        """Extract raw C function pointer from a scipy Cython capsule.

        scipy.linalg.cython_lapack.__pyx_capi__ stores each LAPACK function as
        a PyCapsule whose name is the C signature string.  We first retrieve
        that name, then pass it back to PyCapsule_GetPointer so the name check
        succeeds.  The returned pointer is directly callable as the LAPACK
        function (scipy's OpenBLAS uses a 'scipy_' namespace prefix internally
        but the Cython module exposes the same ABI through these capsules).
        """
        cap = _capi[name]
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        cap_name = ctypes.pythonapi.PyCapsule_GetName(cap)

        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [
            ctypes.py_object, ctypes.c_char_p
        ]
        addr = ctypes.pythonapi.PyCapsule_GetPointer(cap, cap_name)
        if not addr:
            raise RuntimeError(f"Could not get function pointer for {name!r}")
        return addr

    _jaxtra.set_lapack_fn_ptrs(
        _ptr("sormqr"),
        _ptr("dormqr"),
        _ptr("cunmqr"),
        _ptr("zunmqr"),
    )

    for platform, targets in _jaxtra.registrations().items():
        for name, capsule, api_version in targets:
            ffi.register_ffi_target(
                name, capsule, platform=platform, api_version=api_version
            )


_register_targets()

# ---------------------------------------------------------------------------
# 2. Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_TO_TARGET: dict[np.dtype, str] = {
    np.dtype("float32"):    "lapack_sormqr_ffi",
    np.dtype("float64"):    "lapack_dormqr_ffi",
    np.dtype("complex64"):  "lapack_cunmqr_ffi",
    np.dtype("complex128"): "lapack_zunmqr_ffi",
}

_SUPPORTED = frozenset(_DTYPE_TO_TARGET)


def _promote(dtype: np.dtype) -> np.dtype:
    """Map any dtype to the nearest supported one."""
    dtype = np.dtype(dtype)
    if dtype in _SUPPORTED:
        return dtype
    if np.issubdtype(dtype, np.complexfloating):
        return np.dtype("complex64") if dtype.itemsize <= 8 else np.dtype("complex128")
    return np.dtype("float32") if dtype.itemsize <= 4 else np.dtype("float64")


# ---------------------------------------------------------------------------
# 3. JAX primitive
# ---------------------------------------------------------------------------

ormqr_p = core.Primitive("jaxtra_ormqr")
ormqr_p.multiple_results = False
# Eager (non-JIT) execution: dispatch through XLA just like JAX's own linalg ops.
ormqr_p.def_impl(partial(dispatch.apply_primitive, ormqr_p))


# Abstract eval — shape/dtype inference.
def _ormqr_abstract(a, tau, c, *, left: bool, transpose: bool):
    out_dtype = _promote(np.dtype(a.dtype))
    return core.ShapedArray(c.shape, out_dtype)


ormqr_p.def_abstract_eval(_ormqr_abstract)


# MLIR (XLA) lowering — dispatches to the registered FFI target.
#
# We specify column-major (Fortran) layouts for the matrix inputs/output so
# XLA reorders the data before calling the C++ kernel.  The kernel then sees
# data in the order LAPACK expects, and the output is also column-major so
# XLA correctly interprets it back as a row-major JAX array.
#
# Layout tuples are minor-to-major (as XLA expects):
#   row-major 2-D (default):  (1, 0)  — last dim varies fastest
#   col-major 2-D (Fortran):  (0, 1)  — first dim varies fastest
#   col-major batched (n-D): (n-3, n-2, n-4, ..., 0)  → inner (0,1) col-major
def _col_major_layout(ndim: int) -> tuple[int, ...]:
    """Column-major (Fortran) layout in XLA minor-to-major convention."""
    if ndim < 2:
        return tuple(range(ndim - 1, -1, -1))  # 1-D: (0,)
    # For matrices: (ndim-2, ndim-1) for inner dims, then batch dims descending.
    return (ndim - 2, ndim - 1) + tuple(range(ndim - 3, -1, -1))


def _ormqr_lowering(ctx, a, tau, c, *, left: bool, transpose: bool):
    (a_aval, tau_aval, c_aval) = ctx.avals_in
    target = _DTYPE_TO_TARGET[np.dtype(a_aval.dtype)]

    op_layouts = [
        _col_major_layout(len(a_aval.shape)),    # a: column-major
        tuple(range(len(tau_aval.shape) - 1, -1, -1)),  # tau: 1-D (0,)
        _col_major_layout(len(c_aval.shape)),    # c: column-major
    ]
    res_layouts = [
        _col_major_layout(len(c_aval.shape)),    # c_out: column-major
    ]

    # operand_output_aliases={2: 0}: XLA may reuse c's buffer for c_out,
    # saving a copy.  CopyIfDiffBuffer in the C++ kernel handles both cases.
    return ffi.ffi_lowering(
        target,
        operand_layouts=op_layouts,
        result_layouts=res_layouts,
        operand_output_aliases={2: 0},
    )(
        ctx, a, tau, c,
        left=left,
        transpose=transpose,
    )


mlir.register_lowering(ormqr_p, _ormqr_lowering)


# vmap / batching rule.
def _ormqr_batching(args, dims, *, left, transpose):
    a, tau, c = args
    da, dtau, dc = dims

    if da is not batching.not_mapped:
        a = batching.moveaxis(a, da, 0)
    if dtau is not batching.not_mapped:
        tau = batching.moveaxis(tau, dtau, 0)
    if dc is not batching.not_mapped:
        c = batching.moveaxis(c, dc, 0)

    # Broadcast any un-batched inputs to the batch size.
    batch_size = next(
        x.shape[0] for x, d in zip((a, tau, c), (da, dtau, dc))
        if d is not batching.not_mapped
    )
    if da is batching.not_mapped:
        a = jnp.broadcast_to(a, (batch_size,) + a.shape)
    if dtau is batching.not_mapped:
        tau = jnp.broadcast_to(tau, (batch_size,) + tau.shape)
    if dc is batching.not_mapped:
        c = jnp.broadcast_to(c, (batch_size,) + c.shape)

    result = ormqr_p.bind(a, tau, c, left=left, transpose=transpose)
    return result, 0


batching.primitive_batchers[ormqr_p] = _ormqr_batching


# ---------------------------------------------------------------------------
# 4. Public function
# ---------------------------------------------------------------------------

def ormqr(
    a: jax.Array,
    taus: jax.Array,
    c: jax.Array,
    *,
    left: bool = True,
    transpose: bool = False,
) -> jax.Array:
    """Apply Q from a compact QR factorisation to matrix ``c``.

    Uses the Householder representation directly — Q is **never materialised**.
    This is faster and more memory-efficient than forming Q explicitly.

    Parameters
    ----------
    a:
        Compact Householder matrix as returned by
        ``jax._src.lax.linalg.geqrf`` — shape ``(..., m, k)`` where ``k``
        is the number of reflectors (``k = min(m, n)`` for an ``m × n``
        input matrix).
    taus:
        Householder scalar factors — shape ``(..., k)``.
    c:
        Matrix to be multiplied — shape ``(..., m, n)`` for *left*
        multiplication or ``(..., n, m)`` for *right* multiplication.
    left:
        ``True`` (default): compute ``Q @ c`` or ``Qᴴ @ c``.
        ``False``: compute ``c @ Q`` or ``c @ Qᴴ``.
    transpose:
        ``False`` (default): use ``Q``.
        ``True``: use the conjugate transpose ``Qᴴ`` (= ``Qᵀ`` for real).

    Returns
    -------
    jax.Array
        Same shape and dtype as ``c``.

    Notes
    -----
    This is a JIT-compiled primitive.  It works inside ``jax.jit``,
    ``jax.vmap``, and automatic differentiation is planned for a future
    release.

    Examples
    --------
    Solve a least-squares problem efficiently:

    >>> import jax.numpy as jnp
    >>> from jax._src.lax.linalg import geqrf
    >>> from jaxtra import ormqr
    >>>
    >>> A = jnp.ones((6, 4), dtype=jnp.float64)
    >>> b = jnp.ones(6, dtype=jnp.float64)
    >>> H, taus = geqrf(A)               # compact QR — taus as JAX arrays
    >>> Qtb = ormqr(H, taus, b[:, None], left=True, transpose=True)  # Qᵀ b
    """
    dtype = _promote(np.dtype(jnp.result_type(a)))
    a    = jnp.asarray(a,    dtype=dtype)
    taus = jnp.asarray(taus, dtype=dtype)
    c    = jnp.asarray(c,    dtype=dtype)
    return ormqr_p.bind(a, taus, c, left=left, transpose=transpose)
