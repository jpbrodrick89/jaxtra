# Contributing to jaxtra

jaxtra bridges the gap between JAX and LAPACK routines that are not yet in
jaxlib proper. Each exposed routine has three layers, plus an optional fourth:

1. **C++ kernel** — a typed XLA FFI handler that calls LAPACK (CPU) or cuSolver (GPU).
2. **Module registration** — exposes the handlers to Python as FFI capsules.
3. **Python primitive** — a JAX `Primitive` with shape rule, Python fallback
   lowering, and CPU/GPU FFI lowering.
4. **scipy wrapper** _(optional)_ — a high-level function in
   `jaxtra/scipy/linalg.py` that mirrors `jax.scipy.linalg`.

The sections below walk through each layer in the order you will touch them
when adding a new routine from scratch.

---

## Rebuilding the C extension

After any C++ change, rebuild and reinstall the extension in-place. The right
command depends on how you manage your environment.

### CPU only (default)

```bash
# uv (recommended):
uv sync --frozen --extra dev --no-build-isolation-package jaxtra --reinstall-package jaxtra

# pip:
pip install -e ".[dev]" --no-build-isolation
```

### CPU + GPU (requires CUDA 12+ toolkit)

```bash
# uv (recommended):
uv sync --frozen --extra dev --extra cuda13 --no-build-isolation-package jaxtra --reinstall-package jaxtra

# pip:
pip install -e ".[dev,cuda13]" --no-build-isolation
```

`--no-build-isolation` / `--no-build-isolation-package` is required because
the CMake build needs `jax` and `jaxlib` (for the XLA FFI headers) to already
be present in the environment.

### Clearing stale CMake cache

If the build picks up stale artifacts, wipe the CMake cache first:

```bash
rm -f CMakeCache.txt cmake_install.cmake build.ninja && rm -rf CMakeFiles _deps
```

then re-run the install command above. `scikit-build-core` with
`editable.mode = "inplace"` (set in `pyproject.toml`) writes `_jaxtra*.so`
(and `_jaxtra_cuda*.so` when built) directly into the source tree, so the
rebuilt extension is picked up immediately.

### Verify the build

```bash
# uv:
uv run --locked python -c "import jaxtra"

# pip:
python -c "import jaxtra"
```

---

## Repository layout

```
jaxlib/                     mirrors the jaxlib/ layout in the JAX repo
  cpu/
    lapack_kernels.h        CPU kernel struct declarations (one per routine)
    lapack_kernels.cc       CPU kernel implementations
    lapack_utils.h          XLA FFI utilities (SplitBatch2D, CopyIfDiffBuffer, …)
    jaxtra_module.cc        CPU nanobind module: initialize() + registrations()
  cuda/
    jaxtra_cuda_module.cc   GPU nanobind module: registrations() for CUDA targets
  ffi_helpers.h             bundled from jaxlib: AllocateWorkspace, FFI_ASSIGN_OR_RETURN, …
  gpu/                      bundled verbatim from jaxlib (linguist-vendored)
    handle_pool.h           RAII cuSolver/cuBLAS handle pool template
    vendor.h                gpuSolver* → cuSolver*/hipSolver* macro map
    gpu_kernel_helpers.h/cc JAX_AS_STATUS, AsStatus overloads
    solver_handle_pool.h/cc SolverHandlePool::Borrow
    solver_interface.h/cc   OrmqrBufferSize<T>, Ormqr<T> cuSolver wrappers
    solver_kernels_ffi.h/cc OrmqrFfi XLA FFI handler
CMakeLists.txt              build system (auto-detects CUDA; JAXTRA_CUDA env var overrides)
jaxtra/
  _src/
    lib/
      lapack.py             CPU extension wrapper (registrations, prepare_lapack_call)
      gpu_solver.py         GPU extension wrapper (registrations)
    lax/
      linalg.py             JAX primitives + lowerings (CPU and GPU)
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

**Step 1:** Define FFI handlers using the macro pattern. The macro wires the
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
  jaxlib/cpu/jaxtra_module.cc
  jaxlib/cpu/lapack_kernels.cc
)
```

Only edit it if you create a new `.cc` source file.

---

## 1d. GPU kernel (`jaxlib/gpu/`)

If your LAPACK routine has a cuSolver equivalent (e.g. `ormqr` → `cusolverDn?ormqr`), you can add a GPU path. The GPU files in `jaxlib/gpu/` mirror `jaxlib/gpu/` from PR #35104 verbatim.

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

See _Rebuilding the C extension → CPU + GPU_ at the top for the build commands.

CMake auto-detects the CUDA toolkit: `_jaxtra_cuda.so` is built when `nvcc`/`CUDAToolkit` is found, skipped silently otherwise. Abseil is fetched automatically via `FetchContent` if not installed system-wide. Override with `JAXTRA_CUDA=ON` to make CUDA required (fail if absent).

---

## 1e. GPU hybrid path (no cuSOLVER equivalent)

Some LAPACK routines have no cuSOLVER counterpart (e.g. `?hetrf` / `?hetrs` —
Hermitian indefinite factorization and solve). For these, jaxtra uses a
**hybrid** kernel: the handler is registered on the CUDA platform, receives
GPU buffer pointers, explicitly copies data device→host, calls LAPACK on the
CPU, then copies results back device. This mirrors the `_hybrid.so` pattern
used by JAX for `geqp3` with MAGMA absent.

### When to use hybrid vs. cuSOLVER directly

| Situation                                                                                                                                  | Approach                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| cuSOLVER has no equivalent at all (e.g. hetrf/hetrs)                                                                                       | **Hybrid** — full D→H→LAPACK→H→D                                                                       |
| cuSOLVER has the routine but with a different integer width or API (e.g. `cusolverDnXsytrs` needs `int64` ipiv but `sytrf` stores `int32`) | **cuSOLVER with conversion** — do the int32→int64 cast in the FFI handler; keep the computation on GPU |
| cuSOLVER has a direct equivalent                                                                                                           | **cuSOLVER directly** — see section 1d                                                                 |

The hybrid path imposes a full PCIe round-trip for the factor matrix and RHS
(O(n²) data). For n ≲ 100 the latency is negligible; for large n prefer
cuSOLVER or MAGMA. The int-conversion approach (middle row) keeps all heavy
compute on the GPU and only adds an O(n) round-trip for the pivot array.

### Files to touch

| File                                  | What to add                                                      |
| ------------------------------------- | ---------------------------------------------------------------- |
| `jaxlib/gpu/hybrid_kernels_ffi.h`     | Template struct + extern instantiation declarations              |
| `jaxlib/gpu/hybrid_kernels_ffi.cc`    | Kernel implementation + explicit instantiations + handler macros |
| `jaxlib/cuda/jaxtra_hybrid_module.cc` | `initialize()` pointer loads + `registrations()` entries         |
| `jaxtra/_src/lax/linalg.py`           | Route the GPU lowering to `cuhybrid_Xname_ffi`                   |

`jaxtra/_src/lib/gpu_hybrid.py` requires **no change** — it already loads
`_jaxtra_hybrid` and calls `initialize()`.

### `hybrid_kernels_ffi.h` — declare the struct

```cpp
template <ffi::DataType dtype>
struct HybridMyKernel {
  using ValueType = ffi::NativeType<dtype>;

  // Match the LAPACK Fortran signature exactly.
  using FnType = void(char* /*uplo*/, int* /*n*/, ValueType* /*a*/,
                      /* ... */, int* /*info*/);

  inline static FnType* fn = nullptr;  // set by initialize()

  // NOTE: take gpuStream_t directly — XLA FFI auto-unwraps PlatformStream<T>
  // before calling the handler; using PlatformStream here is a compile error.
  static ffi::Error Kernel(gpuStream_t stream,
                            ffi::Buffer<dtype> a, bool lower,
                            ffi::ResultBuffer<dtype> a_out,
                            ffi::ResultBuffer<ffi::DataType::S32> ipiv_out);
};

extern template struct HybridMyKernel<ffi::DataType::C64>;
extern template struct HybridMyKernel<ffi::DataType::C128>;

XLA_FFI_DECLARE_HANDLER_SYMBOL(cuhybrid_cmykernel_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(cuhybrid_zmykernel_ffi);
```

### `hybrid_kernels_ffi.cc` — implement the kernel

The standard pattern is: **sync → D→H copy → sync → LAPACK loop → H→D copy → sync**.
The final sync keeps host buffers (`std::vector`) alive until the async
H→D copy completes.

```cpp
template <ffi::DataType dtype>
ffi::Error HybridMyKernel<dtype>::Kernel(
    gpuStream_t stream,
    ffi::Buffer<dtype> a, bool lower,
    ffi::ResultBuffer<dtype> a_out,
    ffi::ResultBuffer<ffi::DataType::S32> ipiv_out) {
  if (fn == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "HybridMyKernel: not initialized");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, n_rows, n_cols]),
                       SplitBatch2D(a.dimensions()));
  // ... workspace query, host buffer allocation ...

  // 1. Ensure prior GPU writes to `a` are visible before D→H copy.
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  // 2. Copy input buffers device → host.
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(h_a.data(), a.typed_data(),
                     batch * a_elems * sizeof(ValueType),
                     gpuMemcpyDeviceToHost, stream));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  // 3. Call LAPACK on the host for each batch element.
  for (int64_t i = 0; i < batch; ++i) {
    fn(/* ... */);
  }

  // 4. Copy results host → device (async on the stream).
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(a_out->typed_data(), h_a.data(),
                     batch * a_elems * sizeof(ValueType),
                     gpuMemcpyHostToDevice, stream));

  // 5. Sync so host buffers (stack-allocated std::vector) stay alive.
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));
  return ffi::Error::Success();
}

// Explicit instantiations
template struct HybridMyKernel<ffi::DataType::C64>;
template struct HybridMyKernel<ffi::DataType::C128>;

// Handler definitions — note PlatformStream in the macro binding; XLA
// unwraps it to gpuStream_t before calling Kernel.
#define JAXTRA_GPU_DEFINE_HYBRID_MYKERNEL(name, dtype)          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                 \
      name, HybridMyKernel<dtype>::Kernel,                       \
      ffi::Ffi::Bind()                                           \
          .Ctx<ffi::PlatformStream<gpuStream_t>>()               \
          .Arg<ffi::Buffer<dtype>>()              /* a */        \
          .Attr<bool>("lower")                                    \
          .Ret<ffi::Buffer<dtype>>()              /* a_out */    \
          .Ret<ffi::Buffer<ffi::DataType::S32>>()) /* ipiv_out */

JAXTRA_GPU_DEFINE_HYBRID_MYKERNEL(cuhybrid_cmykernel_ffi, ffi::DataType::C64);
JAXTRA_GPU_DEFINE_HYBRID_MYKERNEL(cuhybrid_zmykernel_ffi, ffi::DataType::C128);
```

### `jaxtra_hybrid_module.cc` — register in the CUDA module

In `initialize()`, load the LAPACK pointer from `scipy.linalg.cython_lapack`:

```cpp
AssignFn<HybridMyKernel<ffi::DataType::C64>>(lapack_ptr("cmykernel"));
AssignFn<HybridMyKernel<ffi::DataType::C128>>(lapack_ptr("zmykernel"));
```

In `registrations()`, append to `gpu_targets`:

```cpp
make_entry("cuhybrid_cmykernel_ffi",
           reinterpret_cast<void*>(cuhybrid_cmykernel_ffi));
make_entry("cuhybrid_zmykernel_ffi",
           reinterpret_cast<void*>(cuhybrid_zmykernel_ffi));
```

Note that `registrations()` returns `{"CUDA": [...]}` (not `{"cpu": [...]}`).

### Python lowering — route to the hybrid target

In the GPU branch of `_mycall_cpu_gpu_lowering` in `linalg.py`:

```python
if target_name_prefix != "cpu":
    # No cuSOLVER equivalent; use hybrid CPU-LAPACK kernel.
    char = "c" if dtype == np.complex64 else "z"
    target_name = f"{target_name_prefix}hybrid_{char}mykernel_ffi"
    rule = _linalg_ffi_lowering(target_name, avals_out=[...],
                                 operand_output_aliases={0: 0})
    return rule(ctx, a, lower=lower)
```

The `target_name_prefix` is `"cu"` for CUDA, so `f"{target_name_prefix}hybrid_cmykernel_ffi"` produces `"cuhybrid_cmykernel_ffi"` — matching the name registered in `jaxtra_hybrid_module.cc`.

### Update `docs/conf.py`

The Sphinx stub list must include the new module so docs can be built without
the compiled `.so`:

```python
for _mod in ("jaxtra._jaxtra", "jaxtra._jaxtra_cuda", "jaxtra._jaxtra_hybrid"):
    _stub = types.ModuleType(_mod)
    _stub.initialize = lambda: None
    _stub.registrations = lambda: {}
    sys.modules[_mod] = _stub
```

### CMakeLists.txt

The `_jaxtra_hybrid` target is already defined inside the
`if(_jaxtra_cuda_build)` block. If you only add new kernels to the existing
`hybrid_kernels_ffi.h/.cc` files no CMake change is needed. Only edit
`CMakeLists.txt` if you create an entirely new `.cc` source file.

---

## 3. Python primitive (`jaxtra/_src/lax/linalg.py`)

Each routine needs four things in `_src/lax/linalg.py`: a public function,
a shape rule, a Python fallback lowering, and a CPU/GPU FFI lowering.

### 3a. Shape rule

The shape rule receives abstract shapes (not values) and must return the output
shape or raise `ValueError` for invalid inputs. Batch dimensions are stripped
by `standard_linalg_primitive`; the rule sees only the core (non-batch)
shapes:

```python
def _mynew_shape_rule(arg1_shape, arg2_shape, *, attr):
    # validate and return output shape
    return arg1_shape
```

### 3b. Python fallback lowering

This runs when no FFI backend is available (e.g., in `jax.grad` tracing,
vmap, or any platform without a registered FFI target). Implement it using
`lax` ops and `control_flow.fori_loop` so that JAX can trace through it:

```python
def _mynew_lowering(arg1, arg2, *, attr):
    # Pure JAX implementation using lax primitives.
    ...
    return result
```

Note: use `from jax._src.lax import lax` (internal) to access private helpers
like `lax._eye`. The public `from jax import lax` does not expose these.

### 3c. CPU/GPU FFI lowering

This routes to the FFI target registered by the C extension (CPU) or to a
cuSolver target (GPU). For GPU, the target name follows the pattern
`{prefix}solver_mynew_ffi` (e.g. `cusolver_mynew_ffi` for CUDA), which must
match the name registered in `jaxtra_cuda_module.cc`. For CPU, the target
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

## 4. scipy wrapper (`jaxtra/scipy/linalg.py`, optional)

If your routine mirrors a `jax.scipy.linalg` function, add a wrapper here
with the full SciPy-compatible signature (handling 1-D inputs, mode strings,
overwrite flags, etc.) and import the primitive from `jaxtra._src.lax.linalg`:

```python
from jaxtra._src.lax.linalg import mynew
```

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

Use `pytest.mark.parametrize` for dtype/shape/flag sweeps. Tolerances:

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

| Step                                                                       | File(s)                                  |
| -------------------------------------------------------------------------- | ---------------------------------------- |
| Declare kernel struct + extern templates                                   | `jaxlib/cpu/lapack_kernels.h`            |
| Implement `GetWorkspaceSize` + `Kernel` + instantiations                   | `jaxlib/cpu/lapack_kernels.cc`           |
| Add macro, `initialize()` assignments, `registrations()` entries           | `jaxlib/cpu/jaxtra_module.cc`            |
| Add new `.cc` source if needed                                             | `CMakeLists.txt`                         |
| Rebuild (see _Rebuilding the C extension_ at top)                          | —                                        |
| Shape rule, Python fallback, CPU/GPU lowering, `standard_linalg_primitive` | `jaxtra/_src/lax/linalg.py`              |
| High-level wrapper                                                         | `jaxtra/scipy/linalg.py` (if applicable) |
| Parametrized correctness + JIT + vmap tests                                | `tests/`                                 |

### GPU path (optional, requires CUDA)

| Step                                                                               | File(s)                             |
| ---------------------------------------------------------------------------------- | ----------------------------------- |
| Declare cuSolver wrapper templates                                                 | `jaxlib/gpu/solver_interface.h`     |
| Implement cuSolver wrappers                                                        | `jaxlib/gpu/solver_interface.cc`    |
| Add `gpusolverDn*` macros                                                          | `jaxlib/gpu/vendor.h`               |
| Declare FFI handler symbol                                                         | `jaxlib/gpu/solver_kernels_ffi.h`   |
| Implement `*Impl`, `*Dispatch`, `XLA_FFI_DEFINE_HANDLER_SYMBOL`                    | `jaxlib/gpu/solver_kernels_ffi.cc`  |
| Register GPU target in `registrations()`                                           | `jaxlib/cuda/jaxtra_cuda_module.cc` |
| Rebuild with `--extra cuda13` (see _Rebuilding the C extension_ at top)            | —                                   |
| GPU target name in `_cpu_gpu_lowering` (already routes via `{prefix}solver_*_ffi`) | `jaxtra/_src/lax/linalg.py`         |
