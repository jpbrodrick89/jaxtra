# Contributing to jaxtra

jaxtra bridges the gap between JAX and LAPACK routines that are not yet in
jaxlib proper.  Each exposed routine has four layers:

1. **C++ kernel** — a typed XLA FFI handler that calls LAPACK (CPU) or cuSolver (GPU).
2. **Module registration** — exposes the handlers to Python as FFI capsules.
3. **Python primitive** — a JAX `Primitive` with shape rule, Python fallback
   lowering, and CPU/GPU FFI lowering.
4. **Public API** — the user-facing Python function and its re-exports.

The sections below walk through each layer in the order you will touch them
when adding a new routine from scratch.

---

## Repository layout

```
csrc/
  lapack_kernels.h          CPU kernel struct declarations (one per routine)
  lapack_kernels.cc         CPU kernel implementations
  lapack_utils.h            XLA FFI utilities (SplitBatch2D, CopyIfDiffBuffer, …)
  jaxtra_module.cc          CPU nanobind module: initialize() + registrations()
  jaxtra_cuda_module.cc     GPU nanobind module: registrations() for CUDA targets
  gpu/
    jaxlib/
      ffi_helpers.h         bundled from jaxlib: AllocateWorkspace, FFI_ASSIGN_OR_RETURN, …
      gpu/
        handle_pool.h       RAII cuSolver/cuBLAS handle pool template
        vendor.h            gpuSolver* → cuSolver*/hipSolver* macro map
        gpu_kernel_helpers.h/cc   JAX_AS_STATUS, AsStatus overloads
        solver_handle_pool.h/cc   SolverHandlePool::Borrow
        solver_interface.h/cc     OrmqrBufferSize<T>, Ormqr<T> cuSolver wrappers
        solver_kernels_ffi.h/cc   OrmqrFfi XLA FFI handler
CMakeLists.txt              build system (auto-detects CUDA; JAXTRA_CUDA env var overrides)
jaxtra/
  _core.py                  JAX primitives + lowerings (CPU and GPU)
  lax/linalg.py             jaxtra.lax.linalg public API
  scipy/linalg.py           jaxtra.scipy.linalg public API
tests/
  test_ormqr.py             pytest test suite
```

---

## 1. C++ kernel (`csrc/`)

### 1a. `lapack_kernels.h` — declare the kernel struct

Create a new template struct following the `OrthogonalQrMultiply` pattern.
The struct holds:

- `FnType` — the raw Fortran LAPACK calling convention.
- `fn` — a static function pointer (set to `nullptr` until `initialize()` is
  called at Python import time).
- `Kernel` — the XLA FFI handler invoked by JAX.
- `GetWorkspaceSize` — performs a LAPACK workspace query.

```cpp
template <ffi::DataType dtype>
struct MyNewKernel {
  using ValueType = ffi::NativeType<dtype>;

  // Match the LAPACK Fortran signature exactly.
  using FnType = void(/* ... LAPACK args ... */);

  inline static FnType* fn = nullptr;

  static ffi::Error Kernel(/* ffi::Buffer args */, /* ffi::ResultBuffer */);
  static int64_t GetWorkspaceSize(/* relevant dims */);
};

// Extern declarations for the four standard dtypes.
extern template struct MyNewKernel<ffi::DataType::F32>;
extern template struct MyNewKernel<ffi::DataType::F64>;
extern template struct MyNewKernel<ffi::DataType::C64>;
extern template struct MyNewKernel<ffi::DataType::C128>;
```

### 1b. `lapack_kernels.cc` — implement the kernel

Implement two methods:

- **`GetWorkspaceSize`**: call the LAPACK routine with `lwork=-1` to query the required workspace size.
- **`Kernel`**: loop over batch slices and call `fn`. Add this line before the call:

```cpp
CopyIfDiffBuffer(c, c_out);  // copies input → output when XLA doesn't alias them
```

Add explicit instantiations at the bottom of the file:

```cpp
template struct MyNewKernel<ffi::DataType::F32>;
template struct MyNewKernel<ffi::DataType::F64>;
template struct MyNewKernel<ffi::DataType::C64>;
template struct MyNewKernel<ffi::DataType::C128>;
```

### 1c. `jaxtra_module.cc` — register the handlers

**Step 1:** Define FFI handlers using the macro pattern.  The macro wires the
XLA FFI typed binding (arguments and attributes) to your `Kernel` function.
Adjust `.Arg<>`, `.Attr<>`, and `.Ret<>` to match your LAPACK routine's
signature:

```cpp
#define JAXTRA_CPU_DEFINE_MYNEW(name, dtype)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                   \
      name, MyNewKernel<dtype>::Kernel,                             \
      ffi::Ffi::Bind()                                             \
          .Arg<ffi::Buffer<dtype>>() /* arg1 */                    \
          /* ... */                                                 \
          .Attr<bool>("some_attr")                                  \
          .Ret<ffi::Buffer<dtype>>())

JAXTRA_CPU_DEFINE_MYNEW(lapack_smynew_ffi, ffi::DataType::F32);
JAXTRA_CPU_DEFINE_MYNEW(lapack_dmynew_ffi, ffi::DataType::F64);
JAXTRA_CPU_DEFINE_MYNEW(lapack_cmynew_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_MYNEW(lapack_zmynew_ffi, ffi::DataType::C128);
```

**Step 2:** Inside `initialize()`, load the function pointers from
`scipy.linalg.cython_lapack`:

```cpp
AssignKernelFn<MyNewKernel<ffi::DataType::F32>>(lapack_ptr("smynew"));
AssignKernelFn<MyNewKernel<ffi::DataType::F64>>(lapack_ptr("dmynew"));
AssignKernelFn<MyNewKernel<ffi::DataType::C64>>(lapack_ptr("cmynew"));
AssignKernelFn<MyNewKernel<ffi::DataType::C128>>(lapack_ptr("zmynew"));
```

**Step 3:** Inside `registrations()`, append the four handlers to
`cpu_targets`:

```cpp
make_entry("lapack_smynew_ffi", reinterpret_cast<void*>(lapack_smynew_ffi));
make_entry("lapack_dmynew_ffi", reinterpret_cast<void*>(lapack_dmynew_ffi));
make_entry("lapack_cmynew_ffi", reinterpret_cast<void*>(lapack_cmynew_ffi));
make_entry("lapack_zmynew_ffi", reinterpret_cast<void*>(lapack_zmynew_ffi));
```

### Does `CMakeLists.txt` need editing?

**For CPU-only additions: no.** CMakeLists already compiles both `.cc` files:

```cmake
nanobind_add_module(_jaxtra MODULE
  csrc/jaxtra_module.cc
  csrc/lapack_kernels.cc
)
```

Only edit it if you create a new `.cc` source file.

---

## 1d. GPU kernel (`csrc/gpu/`)

If your LAPACK routine has a cuSolver equivalent (e.g. `ormqr` → `cusolverDn?ormqr`), you can add a GPU path. The GPU files in `csrc/gpu/jaxlib/gpu/` mirror `jaxlib/gpu/` from PR #35104 verbatim.

### `solver_interface.h` — declare the cuSolver wrapper

Add the buffer-size and compute function declarations using the `JAX_GPU_SOLVER_EXPAND_DEFINITION` macro pattern already in the file:

```cpp
// Householder multiply: mynew
#define JAX_GPU_SOLVER_MyNewBufferSize_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, /* ... dims ... */
JAX_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<int>, MyNewBufferSize);
#undef JAX_GPU_SOLVER_MyNewBufferSize_ARGS

#define JAX_GPU_SOLVER_MyNew_ARGS(Type, ...) \
  gpusolverDnHandle_t handle, /* ... all args ... */, int *info
JAX_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, MyNew);
#undef JAX_GPU_SOLVER_MyNew_ARGS
```

### `solver_interface.cc` — implement the cuSolver wrapper

Add a `JAX_GPU_DEFINE_MYNEW` macro block and instantiate it for all four types:

```cpp
#define JAX_GPU_DEFINE_MYNEW(Type, Name)           \
  template <>                                      \
  absl::StatusOr<int> MyNewBufferSize<Type>(...) { \
    int lwork;                                     \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(             \
        Name##_bufferSize(..., &lwork)));           \
    return lwork;                                  \
  }                                                \
  template <>                                      \
  absl::Status MyNew<Type>(...) {                  \
    return JAX_AS_STATUS(Name(...));               \
  }

JAX_GPU_DEFINE_MYNEW(float,           gpusolverDnSmynew);
JAX_GPU_DEFINE_MYNEW(double,          gpusolverDnDmynew);
JAX_GPU_DEFINE_MYNEW(gpuComplex,      gpusolverDnCmynew);
JAX_GPU_DEFINE_MYNEW(gpuDoubleComplex, gpusolverDnZmynew);
#undef JAX_GPU_DEFINE_MYNEW
```

Add the corresponding `gpusolverDnS/D/C/Zmynew` macro definitions to `vendor.h` (CUDA section ~line 200, HIP section ~line 640):

```cpp
// CUDA (vendor.h ~line 200)
#define gpusolverDnSmynew cusolverDnSmynew
#define gpusolverDnDmynew cusolverDnDmynew
// etc.

// HIP (vendor.h ~line 640)
#define gpusolverDnSmynew hipsolverSmynew
// etc.
```

### `solver_kernels_ffi.h` — declare the FFI handler symbol

```cpp
XLA_FFI_DECLARE_HANDLER_SYMBOL(MyNewFfi);
```

### `solver_kernels_ffi.cc` — implement the FFI handler

Add `MyNewImpl<T>`, `MyNewDispatch`, and `XLA_FFI_DEFINE_HANDLER_SYMBOL` following the `OrmqrImpl` / `OrmqrDispatch` / `OrmqrFfi` pattern already in the file.

### `jaxtra_cuda_module.cc` — register the GPU target

Add one `make_entry` call in `registrations()`:

```cpp
make_entry(JAX_GPU_PREFIX "solver_mynew_ffi",
           reinterpret_cast<void*>(MyNewFfi));
```

### Building the GPU extension

```bash
pip install -e ".[gpu]" --no-build-isolation
```

CMake auto-detects the CUDA toolkit: `_jaxtra_cuda.so` is built when `nvcc`/`CUDAToolkit` is found, skipped silently otherwise. Abseil is fetched automatically via `FetchContent` if not installed system-wide. Override with `JAXTRA_CUDA=ON pip install ...` to make CUDA required (fail if absent).

---

## 2. Rebuilding the C extension

After any C++ change, rebuild and reinstall the extension in-place.  The right
command depends on how you manage your environment.

### CPU only (default)

```bash
pip install -e . --no-build-isolation
# or with uv:
uv sync --reinstall-package jaxtra --no-build-isolation
```

### CPU + GPU (requires CUDA toolkit)

```bash
pip install -e ".[gpu]" --no-build-isolation
```

`--no-build-isolation` is required because the CMake build needs `jax` and
`jaxlib` (for the XLA FFI headers) to already be present in the environment.

If that still pulls stale build artifacts, wipe the CMake cache first:

```bash
rm -rf _skbuild build
pip install -e . --no-build-isolation          # or: pip install -e ".[gpu]" --no-build-isolation for GPU
```

`scikit-build-core` with `editable.mode = "inplace"` (set in `pyproject.toml`)
writes `_jaxtra*.so` (and `_jaxtra_cuda*.so` when built) directly into the
source tree, so the rebuilt extension is picked up immediately.

Verify both extensions load cleanly:

```bash
python -c "import jaxtra"
```

---

## 3. Python primitive (`jaxtra/_core.py`)

Each routine needs four things in `_core.py`: a public function, a shape rule,
a Python fallback lowering, and a CPU/GPU FFI lowering.

### 3a. Shape rule

The shape rule receives abstract shapes (not values) and must return the output
shape or raise `ValueError` for invalid inputs.  Batch dimensions are stripped
by `standard_linalg_primitive`; the rule sees only the core (non-batch)
shapes:

```python
def _mynew_shape_rule(arg1_shape, arg2_shape, *, attr):
    # validate and return output shape
    return arg1_shape
```

### 3b. Python fallback lowering

This runs when no FFI backend is available (e.g., in `jax.grad` tracing,
vmap, or any platform without a registered FFI target).  Implement it using
`lax` ops and `control_flow.fori_loop` so that JAX can trace through it:

```python
def _mynew_lowering(arg1, arg2, *, attr):
    # Pure JAX implementation using lax primitives.
    ...
    return result
```

Note: use `from jax._src.lax import lax` (internal) to access private helpers
like `lax._eye`.  The public `from jax import lax` does not expose these.

### 3c. CPU/GPU FFI lowering

This routes to the FFI target registered by the C extension (CPU) or to a
cuSolver target (GPU).  For GPU, the target name follows the pattern
`{prefix}solver_mynew_ffi` (e.g. `cusolver_mynew_ffi` for CUDA), which must
match the name registered in `jaxtra_cuda_module.cc`.  For CPU, the target
name is resolved by `lapack.prepare_lapack_call`, which maps dtype to the
correct LAPACK prefix:

```python
def _mynew_cpu_gpu_lowering(ctx, arg1, arg2, *, attr,
                             target_name_prefix: str):
    a_aval, _ = ctx.avals_in
    if target_name_prefix == "cpu":
        dtype = a_aval.dtype
        # prefix is "un" for complex, "or" for real — adjust for your routine.
        prefix = "un" if dtypes.issubdtype(dtype, np.complexfloating) else "or"
        target_name = lapack.prepare_lapack_call(f"{prefix}mynew_ffi", dtype)
    else:
        target_name = f"{target_name_prefix}solver_mynew_ffi"
    rule = _linalg_ffi_lowering(target_name, operand_output_aliases={1: 0})
    return rule(ctx, arg1, arg2, attr=attr)
```

The `operand_output_aliases` dict maps output index → input index for buffers
that can be aliased in-place (telling XLA it may reuse the input buffer for the
output, which `CopyIfDiffBuffer` handles on the C++ side).

### 3d. Register the primitive

```python
mynew_p = standard_linalg_primitive(
    (_float | _complex, _float | _complex),  # one entry per operand
    (2, 2),                                  # rank of each operand
    _mynew_shape_rule, "mynew")

mlir.register_lowering(mynew_p, mlir.lower_fun(
    _mynew_lowering, multiple_results=False))

register_cpu_gpu_lowering(mynew_p, _mynew_cpu_gpu_lowering)
```

`standard_linalg_primitive` wires `expand_dims_batcher` automatically, giving
free batch dimension support.

### 3e. Public function

Add the user-facing wrapper that calls `ormqr_p.bind(...)`:

```python
def mynew(arg1: ArrayLike, arg2: ArrayLike, *, attr: bool = True) -> Array:
    """Docstring."""
    arg1, arg2 = core.standard_insert_pvary(arg1, arg2)
    return mynew_p.bind(arg1, arg2, attr=attr)
```

---

## 4. Public API (`jaxtra/lax/linalg.py`)

Re-export the new function and primitive from `jaxtra.lax.linalg`:

```python
from jaxtra._core import mynew, mynew_p  # noqa: F401

__all__ = [..., "mynew", "mynew_p"]
```

If your function mirrors a `jax.scipy.linalg` routine, add it to
`jaxtra/scipy/linalg.py` with the full SciPy-compatible signature (handling
1-D inputs, mode strings, overwrite flags, etc.).

---

## 5. Tests (`tests/`)

Add tests that cover:

- **Correctness**: compare against a reference built from JAX's
  `householder_product` or `jax.scipy.linalg`, not from NumPy/SciPy, so that
  the test exercises the same dtype promotion path.
- **All dtypes**: `float32`, `float64`, `complex64`, `complex128`.
- **Batched inputs**: at least one batched shape alongside the 2-D cases.
- **JIT**: wrap the call in `@jax.jit` and compare to the eager result.
- **vmap**: vectorise over a batch dimension and compare slice-by-slice.

Use `pytest.mark.parametrize` for dtype/shape/flag sweeps.  Tolerances:

```python
tol = {np.float32: 1e-4, np.complex64: 1e-4,
       np.float64: 1e-10, np.complex128: 1e-10}[dtype]
np.testing.assert_allclose(result, expected, rtol=tol, atol=tol)
```

Run the full suite with:

```bash
pytest tests/ -v
```

---

## Summary checklist

### CPU path (always required)

| Step | File(s) |
|------|---------|
| Declare kernel struct + extern templates | `csrc/lapack_kernels.h` |
| Implement `GetWorkspaceSize` + `Kernel` + instantiations | `csrc/lapack_kernels.cc` |
| Add macro, `initialize()` assignments, `registrations()` entries | `csrc/jaxtra_module.cc` |
| Add new `.cc` source if needed | `CMakeLists.txt` |
| Rebuild: `pip install -e . --no-build-isolation` | — |
| Shape rule, Python fallback, CPU/GPU lowering, `standard_linalg_primitive` | `jaxtra/_core.py` |
| Re-export | `jaxtra/lax/linalg.py` |
| High-level wrapper | `jaxtra/scipy/linalg.py` (if applicable) |
| Parametrized correctness + JIT + vmap tests | `tests/` |

### GPU path (optional, requires CUDA)

| Step | File(s) |
|------|---------|
| Declare cuSolver wrapper templates | `csrc/gpu/jaxlib/gpu/solver_interface.h` |
| Implement cuSolver wrappers | `csrc/gpu/jaxlib/gpu/solver_interface.cc` |
| Add `gpusolverDn*` macros | `csrc/gpu/jaxlib/gpu/vendor.h` |
| Declare FFI handler symbol | `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.h` |
| Implement `*Impl`, `*Dispatch`, `XLA_FFI_DEFINE_HANDLER_SYMBOL` | `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.cc` |
| Register GPU target in `registrations()` | `csrc/jaxtra_cuda_module.cc` |
| Rebuild: `pip install -e ".[gpu]" --no-build-isolation` | — |
| GPU target name in `_cpu_gpu_lowering` (already routes via `{prefix}solver_*_ffi`) | `jaxtra/_core.py` |
