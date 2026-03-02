# Upstreaming a jaxtra routine to JAX

Once a routine has been battle-tested in the wild and explicitly requested by users via a JAX GitHub issue, it is ready to be proposed for inclusion in JAX proper. This document explains how to do that. The appendix uses `ormqr` / `qr_multiply` (PR #35104) as a concrete worked example.

---

## How jaxtra maps onto jaxlib

jaxtra is deliberately written to mirror the structure of jaxlib. The kernel code, primitive registration, and lowering rules are all written in the same style as their jaxlib counterparts. The differences are structural, not algorithmic:

| Concern                                   | jaxtra                                                                                     | jaxlib (upstream JAX)                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Build system                              | CMake + scikit-build-core                                                                  | Bazel                                                                            |
| LAPACK function pointers                  | Loaded in our `initialize()` via `scipy.linalg.cython_lapack` capsules                     | Loaded in jaxlib's `GetLapackKernelsFromScipy()` — same mechanism, same source   |
| CPU utility helpers (`SplitBatch2D` etc.) | Reimplemented in `csrc/lapack_utils.h` (public XLA FFI API only)                           | Identical helpers already exist in jaxlib's internal headers; just omit our copy |
| C++ namespace                             | `namespace jaxtra` (CPU); `namespace jax::JAX_GPU_NAMESPACE` (GPU, matches jaxlib exactly) | `namespace jax`                                                                  |
| CPU module + handler registration         | `csrc/jaxtra_module.cc`                                                                    | Added to `jaxlib/cpu/cpu_kernels.cc`                                             |
| GPU cuSolver wrappers                     | `csrc/gpu/jaxlib/gpu/solver_interface.{h,cc}`                                              | `jaxlib/gpu/solver_interface.{h,cc}`                                             |
| GPU FFI handler                           | `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.{h,cc}`                                            | `jaxlib/gpu/solver_kernels_ffi.{h,cc}`                                           |
| GPU vendor abstraction                    | `csrc/gpu/jaxlib/gpu/vendor.h` (bundled copy)                                              | `jaxlib/gpu/vendor.h`                                                            |
| GPU module registration                   | `csrc/jaxtra_cuda_module.cc`                                                               | Added to `jaxlib/gpu/solver.cc` + `jaxlib/gpu/gpu_kernels.cc`                    |
| GPU helper dependencies                   | Bundled in `csrc/gpu/jaxlib/` (`ffi_helpers.h`, `handle_pool.h`, etc.)                     | Internal jaxlib headers already in scope                                         |
| FFI target registration                   | `jaxtra/_src/lib/lapack.py`, `jaxtra/_src/lib/gpu_solver.py` (jaxtra scaffolding only)     | Done inside jaxlib at import time; omit entirely                                 |
| JAX primitive                             | `jaxtra/_src/lax/linalg.py`                                                                | `jax/_src/lax/linalg.py`                                                         |
| scipy-level public API                    | `jaxtra/scipy/linalg.py`                                                                   | `jax/_src/scipy/linalg.py` + re-export in `jax/scipy/linalg.py`                  |

---

## Selecting what to upstream

A single PR should contain exactly one routine (or one tightly related group, e.g. a LAPACK function and its scipy-level wrapper). Do not bundle unrelated routines. The PR description should link to the JAX GitHub issue that requested the feature.

---

## Step 1 — Port the C++ kernel

### 1a. `jaxlib/cpu/lapack_kernels.h`

Copy the kernel struct from `csrc/lapack_kernels.h` verbatim, then make two changes:

**1. Change the namespace.**

```diff
-namespace jaxtra {
+namespace jax {
```

**2. Remove the `lapack_utils.h` include.**

`SplitBatch2D`, `MaybeCastNoOverflow`, `CopyIfDiffBuffer`, and `FFI_ASSIGN_OR_RETURN` are already provided by jaxlib's internal headers. Do not include `lapack_utils.h`; its equivalents are already in scope in `lapack_kernels.cc`.

Everything else — the `FnType` typedef, the `fn` static pointer, `Kernel`, `GetWorkspaceSize`, and the `extern template` declarations — is copied unchanged.

---

### 1b. `jaxlib/cpu/lapack_kernels.cc`

Copy the implementation from `csrc/lapack_kernels.cc` verbatim, then:

**1. Change the namespace** (`namespace jaxtra` → `namespace jax`).

**2. Update the include list.** Remove `#include "lapack_utils.h"`. The file already includes the jaxlib headers that provide the same utilities.

The `GetWorkspaceSize` and `Kernel` function bodies are identical. The explicit instantiations at the bottom are identical.

---

### 1c. `jaxlib/cpu/cpu_kernels.cc` — handler definitions and registration

In jaxtra, `csrc/jaxtra_module.cc` does three things: it defines the XLA FFI handler symbols, it assigns the LAPACK function pointers in `initialize()`, and it returns the registrations dict. In jaxlib, the same three things are folded into the existing `cpu_kernels.cc`:

**Handler symbols.** Add the macro invocations from jaxtra's `JAXTRA_CPU_DEFINE_<ROUTINE>` block. Rename the macro prefix to `JAX_CPU_DEFINE_<ROUTINE>`; the body is identical. Instantiate the macro once per dtype (F32, F64, C64, C128 for a full-precision LAPACK routine).

**Function pointer assignment.** Find `GetLapackKernelsFromScipy()` (the equivalent of jaxtra's `initialize()`). Add one `AssignKernelFn<Kernel<dtype>>(lapack_ptr("<lapack_name>"))` call per dtype alongside the existing ones.

**Handler registration.** Find the block where existing LAPACK handlers are appended to the registration list (equivalent of jaxtra's `make_entry(...)` calls). Add one `make_entry("lapack_<name>_ffi", reinterpret_cast<void*>(lapack_<name>_ffi))` per dtype.

---

## Step 2 — Port the GPU kernel (if applicable)

If the routine has a cuSolver equivalent, port the GPU files from
`csrc/gpu/jaxlib/gpu/` into the corresponding files in `jaxlib/gpu/`.

### 2a. `jaxlib/gpu/solver_interface.h` and `solver_interface.cc`

Copy the `JAX_GPU_SOLVER_<Routine>BufferSize_ARGS` / `JAX_GPU_SOLVER_<Routine>_ARGS`
blocks from `csrc/gpu/jaxlib/gpu/solver_interface.h` verbatim. Copy the
`JAX_GPU_DEFINE_<ROUTINE>` macro block and its four instantiations from
`csrc/gpu/jaxlib/gpu/solver_interface.cc` verbatim.

No include or namespace changes are needed — jaxtra's GPU files already use
`namespace jax::JAX_GPU_NAMESPACE` and the same include paths as jaxlib.

### 2b. `jaxlib/gpu/vendor.h`

Add the `gpusolverDn<Type><routine>` and `gpusolverDn<Type><routine>_bufferSize`
defines from `csrc/gpu/jaxlib/gpu/vendor.h` into the CUDA section (~line 200)
and the HIP section (~line 640) of jaxlib's `vendor.h`.

### 2c. `jaxlib/gpu/solver_kernels_ffi.h` and `solver_kernels_ffi.cc`

Add `XLA_FFI_DECLARE_HANDLER_SYMBOL(<Routine>Ffi)` to the header.

Copy the `<Routine>Impl`, `<Routine>Dispatch`, and `XLA_FFI_DEFINE_HANDLER_SYMBOL`
blocks from `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.cc` verbatim into jaxlib's
`solver_kernels_ffi.cc`. The only includes needed (`jaxlib/ffi_helpers.h`,
`jaxlib/gpu/solver_handle_pool.h`, etc.) are already present in the file.

### 2d. `jaxlib/gpu/solver.cc`

Add one line to `Registrations()`:

```cpp
dict[JAX_GPU_PREFIX "solver_<routine>_ffi"] = EncapsulateFfiHandler(<Routine>Ffi);
```

### 2e. `jaxlib/gpu/gpu_kernels.cc`

Add one `XLA_FFI_REGISTER_HANDLER` call (for static registration when the GPU
module is loaded without Python):

```cpp
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_<routine>_ffi", "CUDA",
                         <Routine>Ffi);
```

---

## Step 3 — Bazel BUILD files

For a routine added entirely within existing source files, no new `BUILD`
entries are required. If you need to add a new `.cc` file (rare), find the
`cc_library` target that compiles the relevant file and add yours alongside it:

```bash
grep -n "lapack_kernels.cc" jaxlib/cpu/BUILD
grep -n "solver_kernels_ffi.cc" jaxlib/gpu/BUILD
```

---

## Step 4 — Build jaxlib from source

```bash
python build/build.py build --wheels=jaxlib
```

This runs the full Bazel build and deposits a wheel in `dist/`. Install it into your development environment:

```bash
pip install dist/jaxlib-*.whl --force-reinstall
```

Verify the extension loads:

```bash
python -c "import jaxlib; print(jaxlib.__version__)"
```

If the build fails on a C++ error, the Bazel output will include the file and line number. The most common issues are:

- Missing `extern template` declarations for the new struct in the header.
- Forgetting to add explicit instantiations at the bottom of the `.cc` file.
- A handler macro that does not match the `Kernel` function's argument list exactly.

---

## Step 5 — Port the Python primitive

### 4a. `jax/_src/lax/linalg.py`

Copy the following from `jaxtra/_src/lax/linalg.py` verbatim into `jax/_src/lax/linalg.py`:

- The public function(s) for the routine
- The shape rule (`_<routine>_shape_rule`)
- The Python fallback lowering (`_<routine>_lowering`)
- The CPU/GPU lowering (`_<routine>_cpu_gpu_lowering`)
- The primitive registration block (`<routine>_p = standard_linalg_primitive(...)` and the `register_lowering` calls)

**Remove entirely:**

- The `register_module_custom_calls(lapack)` / `register_module_custom_calls(gpu_solver)`
  calls and the `from jaxtra._src.lib import lapack, gpu_solver` import — jaxlib
  already calls `register_module_custom_calls` for its own modules at import time.
- The `from jaxtra._src.lib import lapack` import in `_ormqr_cpu_gpu_lowering` —
  replace with the `lapack` module already in scope in `jax/_src/lax/linalg.py`
  (imported from `jaxlib`).

No other import changes are needed. `_src/lax/linalg.py` already imports from
`jax._src.lax.linalg` (its destination file), `jax._src.lax.lax`, etc. — the
same paths used throughout `linalg.py`.

Add the new public names to `__all__` at the top of the file.

---

### 4b. `jax/_src/scipy/linalg.py` (if the routine has a scipy-level wrapper)

Copy the wrapper from `jaxtra/scipy/linalg.py` into `jax/_src/scipy/linalg.py`. Update imports to point at `jax._src.lax.linalg` rather than `jaxtra._src.lax.linalg`. Add the new symbol to the re-export list in `jax/scipy/linalg.py`.

---

## Step 6 — Tests

Copy the relevant test file(s) from `tests/` into `tests/` inside the JAX repo. Update the import lines:

```python
# Before (jaxtra)
from jaxtra._src.lax.linalg import <routine>
import jaxtra.scipy.linalg as jsla

# After (JAX)
from jax._src.lax.linalg import <routine>
import jax.scipy.linalg as jsla
```

Run locally before opening the PR:

```bash
pytest tests/<your_test_file>.py -v
```

The parametrised dtype/shape/JIT/vmap coverage in the jaxtra test suite is sufficient for a JAX PR; do not strip it down.

---

## Appendix: worked example — `ormqr` / `qr_multiply` (PR #35104)

### File-by-file map

**CPU path**

| jaxtra                                   | JAX / jaxlib                   | Changes required                                                                                                              |
| ---------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `csrc/lapack_kernels.h`                  | `jaxlib/cpu/lapack_kernels.h`  | Namespace only                                                                                                                |
| `csrc/lapack_kernels.cc`                 | `jaxlib/cpu/lapack_kernels.cc` | Namespace; drop `lapack_utils.h` include                                                                                      |
| `csrc/jaxtra_module.cc` (handler macros) | `jaxlib/cpu/cpu_kernels.cc`    | Rename macro; fold `initialize()` into `GetLapackKernelsFromScipy()`; fold `registrations()` into existing registration block |

**GPU path**

| jaxtra                                      | JAX / jaxlib                         | Changes required                                                                                          |
| ------------------------------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `csrc/gpu/jaxlib/gpu/solver_interface.h`    | `jaxlib/gpu/solver_interface.h`      | None — copy the `OrmqrBufferSize` / `Ormqr` macro blocks verbatim                                         |
| `csrc/gpu/jaxlib/gpu/solver_interface.cc`   | `jaxlib/gpu/solver_interface.cc`     | None — copy the `JAX_GPU_DEFINE_ORMQR` block verbatim                                                     |
| `csrc/gpu/jaxlib/gpu/vendor.h`              | `jaxlib/gpu/vendor.h`                | Copy the `gpusolverDn?ormqr` / `gpusolverDn?ormqr_bufferSize` defines into both the CUDA and HIP sections |
| `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.h`  | `jaxlib/gpu/solver_kernels_ffi.h`    | Add `XLA_FFI_DECLARE_HANDLER_SYMBOL(OrmqrFfi)`                                                            |
| `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.cc` | `jaxlib/gpu/solver_kernels_ffi.cc`   | Copy `OrmqrImpl`, `OrmqrDispatch`, `XLA_FFI_DEFINE_HANDLER_SYMBOL(OrmqrFfi, …)` verbatim                  |
| `csrc/jaxtra_cuda_module.cc` (ormqr entry)  | `jaxlib/gpu/solver.cc`               | Add `dict[JAX_GPU_PREFIX "solver_ormqr_ffi"] = EncapsulateFfiHandler(OrmqrFfi)`                           |
| `csrc/jaxtra_cuda_module.cc` (ormqr entry)  | `jaxlib/gpu/gpu_kernels.cc`          | Add `XLA_FFI_REGISTER_HANDLER(…, "cusolver_ormqr_ffi", "CUDA", OrmqrFfi)`                                 |
| `csrc/gpu/jaxlib/` (bundled helpers)        | Already in jaxlib's internal headers | Drop entirely — jaxlib already has `ffi_helpers.h`, `handle_pool.h`, etc.                                 |

**Python / tests**

| jaxtra                                              | JAX / jaxlib                                       | Changes required                                                                                              |
| --------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `jaxtra/_src/lax/linalg.py` (primitive + lowerings) | `jax/_src/lax/linalg.py`                           | Drop `register_module_custom_calls` calls + lib imports; use jaxlib's `lapack` module; add names to `__all__` |
| `jaxtra/scipy/linalg.py` (`qr_multiply`)            | `jax/_src/scipy/linalg.py` + `jax/scipy/linalg.py` | Update imports; add to re-export list                                                                         |
| `tests/test_ormqr.py`                               | `tests/lax_scipy_linalg_test.py` (or new file)     | Update imports                                                                                                |

### Step 1c detail — `cpu_kernels.cc` additions

**Handler symbols:**

```cpp
#define JAX_CPU_DEFINE_ORMQR(name, dtype)                        \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                  \
      name, OrthogonalQrMultiply<dtype>::Kernel,                  \
      ffi::Ffi::Bind()                                            \
          .Arg<ffi::Buffer<dtype>>()   /* a */                    \
          .Arg<ffi::Buffer<dtype>>()   /* tau */                  \
          .Arg<ffi::Buffer<dtype>>()   /* c */                    \
          .Attr<bool>("left")                                     \
          .Attr<bool>("transpose")                                \
          .Ret<ffi::Buffer<dtype>>())  /* c_out */

JAX_CPU_DEFINE_ORMQR(lapack_sormqr_ffi, ffi::DataType::F32);
JAX_CPU_DEFINE_ORMQR(lapack_dormqr_ffi, ffi::DataType::F64);
JAX_CPU_DEFINE_ORMQR(lapack_cunmqr_ffi, ffi::DataType::C64);
JAX_CPU_DEFINE_ORMQR(lapack_zunmqr_ffi, ffi::DataType::C128);
```

**Function pointer assignment** (inside `GetLapackKernelsFromScipy()`):

```cpp
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32>>(lapack_ptr("sormqr"));
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64>>(lapack_ptr("dormqr"));
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64>>(lapack_ptr("cunmqr"));
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128>>(lapack_ptr("zunmqr"));
```

**Handler registration:**

```cpp
make_entry("lapack_sormqr_ffi", reinterpret_cast<void*>(lapack_sormqr_ffi));
make_entry("lapack_dormqr_ffi", reinterpret_cast<void*>(lapack_dormqr_ffi));
make_entry("lapack_cunmqr_ffi", reinterpret_cast<void*>(lapack_cunmqr_ffi));
make_entry("lapack_zunmqr_ffi", reinterpret_cast<void*>(lapack_zunmqr_ffi));
```

### Step 2 detail — GPU additions for `ormqr`

**`solver_interface.cc`** — the `JAX_GPU_DEFINE_ORMQR` block (verbatim from `csrc/gpu/jaxlib/gpu/solver_interface.cc`) instantiates `OrmqrBufferSize<T>` and `Ormqr<T>` for `float`, `double`, `gpuComplex`, and `gpuDoubleComplex` by wrapping `gpusolverDnSormqr` / `gpusolverDnDormqr` / `gpusolverDnCunmqr` / `gpusolverDnZunmqr`.

**`vendor.h`** — eight new `#define` lines (CUDA section and HIP section):

```cpp
// CUDA (~line 201)
#define gpusolverDnSormqr cusolverDnSormqr
#define gpusolverDnDormqr cusolverDnDormqr
#define gpusolverDnCunmqr cusolverDnCunmqr
#define gpusolverDnZunmqr cusolverDnZunmqr
#define gpusolverDnSormqr_bufferSize cusolverDnSormqr_bufferSize
#define gpusolverDnDormqr_bufferSize cusolverDnDormqr_bufferSize
#define gpusolverDnCunmqr_bufferSize cusolverDnCunmqr_bufferSize
#define gpusolverDnZunmqr_bufferSize cusolverDnZunmqr_bufferSize

// HIP (~line 638)
#define gpusolverDnSormqr hipsolverSormqr
#define gpusolverDnDormqr hipsolverDormqr
#define gpusolverDnCunmqr hipsolverCunmqr
#define gpusolverDnZunmqr hipsolverZunmqr
#define gpusolverDnSormqr_bufferSize hipsolverSormqr_bufferSize
#define gpusolverDnDormqr_bufferSize hipsolverDormqr_bufferSize
#define gpusolverDnCunmqr_bufferSize hipsolverCunmqr_bufferSize
#define gpusolverDnZunmqr_bufferSize hipsolverZunmqr_bufferSize
```

**`solver_kernels_ffi.cc`** — the full `OrmqrImpl` / `OrmqrDispatch` / `XLA_FFI_DEFINE_HANDLER_SYMBOL(OrmqrFfi, …)` block is verbatim from `csrc/gpu/jaxlib/gpu/solver_kernels_ffi.cc`.

**`solver.cc`** — one new line in `Registrations()`:

```cpp
dict[JAX_GPU_PREFIX "solver_ormqr_ffi"] = EncapsulateFfiHandler(OrmqrFfi);
```

**`gpu_kernels.cc`** — one new `XLA_FFI_REGISTER_HANDLER` call:

```cpp
XLA_FFI_REGISTER_HANDLER(XLA_FFI_GetApi(), "cusolver_ormqr_ffi", "CUDA",
                         OrmqrFfi);
```
