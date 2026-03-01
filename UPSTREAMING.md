# Upstreaming a jaxtra routine to JAX

Once a routine has been battle-tested in the wild and explicitly requested by users via a JAX GitHub issue, it is ready to be proposed for inclusion in JAX proper. This document explains how to do that. It is concrete enough to reconstruct PR #35104 (`ormqr` / `qr_multiply`) step by step, and is intended to generalise to any future routine exposed through jaxtra.

---

## How jaxtra maps onto jaxlib

jaxtra is deliberately written to mirror the structure of jaxlib. The kernel code, primitive registration, and lowering rules are all written in the same style as their jaxlib counterparts. The differences are structural, not algorithmic:

| Concern | jaxtra | jaxlib (upstream JAX) |
|---|---|---|
| Build system | CMake + scikit-build-core | Bazel |
| LAPACK function pointers | Loaded in our `initialize()` via `scipy.linalg.cython_lapack` capsules | Loaded in jaxlib's `GetLapackKernelsFromScipy()` — same mechanism, same source |
| Utility helpers (`SplitBatch2D` etc.) | Reimplemented in `csrc/lapack_utils.h` (public XLA FFI API only) | Identical helpers already exist in jaxlib's internal headers; just omit our copy |
| C++ namespace | `namespace jaxtra` | `namespace jax` |
| Module + handler registration | `csrc/jaxtra_module.cc` | Added to `jaxlib/cpu/cpu_kernels.cc` |
| FFI target registration | `_core.py` `_load_extension()` block | Done inside jaxlib at import time; omit entirely |
| JAX primitive | `jaxtra/_core.py` | `jax/_src/lax/linalg.py` |
| scipy-level public API | `jaxtra/scipy/linalg.py` | `jax/_src/scipy/linalg.py` + re-export in `jax/scipy/linalg.py` |

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

In jaxtra, `csrc/jaxtra_module.cc` does three things: it defines the XLA FFI handler symbols, it assigns the LAPACK function pointers in `initialize()`, and it returns the registrations dict. In jaxlib, the same three things are split across the existing `cpu_kernels.cc`:

**Handler symbols.** Add the macro invocations from jaxtra's `JAXTRA_DEFINE_ORMQR` block. Rename the macro to match jaxlib's convention (e.g. `JAX_CPU_DEFINE_ORMQR`); the body is identical:

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

**Function pointer assignment.** Find the existing `GetLapackKernelsFromScipy()` function (the direct equivalent of jaxtra's `initialize()`). Add the four `AssignKernelFn` calls alongside the existing ones:

```cpp
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32>>(lapack_ptr("sormqr"));
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64>>(lapack_ptr("dormqr"));
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64>>(lapack_ptr("cunmqr"));
AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128>>(lapack_ptr("zunmqr"));
```

**Handler registration.** Find the block where existing LAPACK handlers are appended to the registration list (the equivalent of jaxtra's `make_entry(...)` calls in `registrations()`). Add four entries using the same pattern:

```cpp
make_entry("lapack_sormqr_ffi", reinterpret_cast<void*>(lapack_sormqr_ffi));
make_entry("lapack_dormqr_ffi", reinterpret_cast<void*>(lapack_dormqr_ffi));
make_entry("lapack_cunmqr_ffi", reinterpret_cast<void*>(lapack_cunmqr_ffi));
make_entry("lapack_zunmqr_ffi", reinterpret_cast<void*>(lapack_zunmqr_ffi));
```

---

## Step 2 — Bazel BUILD files

For a routine added entirely within existing source files (`lapack_kernels.h`, `lapack_kernels.cc`, `cpu_kernels.cc`), no new `BUILD` entries are required — those files are already listed in the existing `cc_library` targets.

If you need to add a new `.cc` file (rare), find the `cc_library` target that compiles `lapack_kernels.cc` in `jaxlib/cpu/BUILD` and add your file alongside it:

```bash
grep -n "lapack_kernels.cc" jaxlib/cpu/BUILD
```

---

## Step 3 — Build jaxlib from source

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

## Step 4 — Port the Python primitive

### 4a. `jax/_src/lax/linalg.py`

Copy the following from `jaxtra/_core.py` verbatim into `jax/_src/lax/linalg.py`:

- The public function (`ormqr`)
- The shape rule (`_ormqr_shape_rule`)
- The Python fallback lowering (`_ormqr_lowering`)
- The CPU/GPU lowering (`_ormqr_cpu_gpu_lowering`)
- The primitive registration block (`ormqr_p = standard_linalg_primitive(...)` and the two `register_lowering` calls)

**Remove entirely:**

- The `_load_extension()` function and everything that calls it — jaxlib loads the C extension automatically.
- The `ffi.register_ffi_target()` loop — jaxlib handles registration internally via `GetLapackKernelsFromScipy()`.

No import changes are needed. `_core.py` already imports from `jax._src.lax.linalg` (its destination file), `jax._src.lax.lax`, etc. — the same paths used throughout `linalg.py`.

Add the new public names to `__all__` at the top of the file.

---

### 4b. `jax/_src/scipy/linalg.py` (if the routine has a scipy-level wrapper)

Copy `qr_multiply` (or the equivalent wrapper) from `jaxtra/scipy/linalg.py` into `jax/_src/scipy/linalg.py`. Update imports to point at `jax._src.lax.linalg` rather than `jaxtra._core`. Add the new symbol to the re-export list in `jax/scipy/linalg.py`.

---

## Step 5 — Tests

Copy `tests/test_ormqr.py` (or the relevant test file) into `tests/` inside the JAX repo. Update the import lines:

```python
# Before (jaxtra)
from jaxtra import ormqr
import jaxtra.scipy.linalg as jsla

# After (JAX)
from jax._src.lax.linalg import ormqr
import jax.scipy.linalg as jsla
```

Run locally before opening the PR:

```bash
pytest tests/<your_test_file>.py -v
```

The parametrised dtype/shape/JIT/vmap coverage in the jaxtra test suite is sufficient for a JAX PR; do not strip it down.

---

## Appendix: file-by-file map for `ormqr` / `qr_multiply` (PR #35104)

| jaxtra | JAX / jaxlib | Changes required |
|---|---|---|
| `csrc/lapack_kernels.h` | `jaxlib/cpu/lapack_kernels.h` | Namespace only |
| `csrc/lapack_kernels.cc` | `jaxlib/cpu/lapack_kernels.cc` | Namespace; drop `lapack_utils.h` include |
| `csrc/jaxtra_module.cc` (handler macros) | `jaxlib/cpu/cpu_kernels.cc` | Rename macro; fold `initialize()` into `GetLapackKernelsFromScipy()`; fold `registrations()` into existing registration block |
| `jaxtra/_core.py` (primitive + lowerings) | `jax/_src/lax/linalg.py` | Drop `_load_extension` block; add names to `__all__` |
| `jaxtra/scipy/linalg.py` (`qr_multiply`) | `jax/_src/scipy/linalg.py` + `jax/scipy/linalg.py` | Update imports; add to re-export list |
| `tests/test_ormqr.py` | `tests/lax_scipy_linalg_test.py` (or new file) | Update imports |
