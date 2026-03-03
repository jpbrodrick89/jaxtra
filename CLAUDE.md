# CLAUDE.md

## What this repo is

jaxtra exposes missing LAPACK routines as proper JAX primitives via XLA's
Foreign Function Interface. Currently: **`ormqr`** (orthogonal QR multiply).

Public entry point: `jaxtra.scipy.linalg.qr_multiply` (mirrors `jax.scipy.linalg`).
Power-user primitive: `from jaxtra._src.lax.linalg import ormqr` (same pattern
as reaching into `jax._src.lax.linalg` directly).

See **README.md** (Quick start, Installation, API overview) for user-facing
documentation. The canonical usage pattern (least-squares via QR) is in the
Quick start section.

---

## Architecture (four layers)

| Layer               | File(s)                                                      | What lives here                                                                                             |
| ------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| C++ kernel          | `csrc/lapack_kernels.{h,cc}`                                 | `OrthogonalQrMultiply<dtype>` — LAPACK function pointer, workspace query, batch loop                        |
| Module registration | `csrc/jaxtra_module.cc`                                      | XLA FFI handler macros, `initialize()` (loads LAPACK pointers from SciPy), `registrations()`                |
| Lib wrappers        | `jaxtra/_src/lib/lapack.py`, `jaxtra/_src/lib/gpu_solver.py` | Extension loading, `registrations()`, `batch_partitionable_targets()`, `prepare_lapack_call()`              |
| JAX primitive       | `jaxtra/_src/lax/linalg.py`                                  | `ormqr_p`, shape rule, Python fallback lowering, CPU/GPU FFI lowering, `register_module_custom_calls` calls |
| Public API          | `jaxtra/scipy/linalg.py`                                     | `qr_multiply` wrapper                                                                                       |

GPU path mirrors the same C++ layers under `csrc/gpu/` and
`csrc/jaxtra_cuda_module.cc`.

---

## Environment setup (do this first)

Before reasoning about any task that involves running Python, tests, linting,
or benchmarks, **immediately** kick off the environment setup in the
background so dependencies are ready by the time you need them:

```bash
# Install all deps (frozen lock file, build jaxtra in-place)
uv sync --frozen --extra test --no-build-isolation-package jaxtra
```

```bash
# Smoke-test — verifies the install succeeded
uv run --locked python -c "from jaxtra.scipy.linalg import qr_multiply; print('ok')"
```

Run both commands **in parallel with your initial exploration** of the task.

**Always** prefix Python commands with `uv run --locked`. Never use bare
`python`, `pytest`, or `pre-commit` — always go through `uv run --locked`.

Examples:

```bash
uv run --locked pytest tests/ -v            # run tests
uv run --locked pre-commit run --all-files  # run linters
uv run --locked python benchmarks/bench_least_squares.py  # benchmarks
```

---

## Common tasks

### Adding a new routine

Follow **CONTRIBUTING.md** top-to-bottom. It walks all four layers with code
templates. Summary checklist is at the end.

### Running tests

```bash
uv run --locked pytest tests/ -v
```

### Rebuilding after C++ changes

Python-only changes (anything under `jaxtra/`) take effect immediately.
After **any** edit to `csrc/`, rebuild before testing. See
**CONTRIBUTING.md § 2. Rebuilding the C extension** for the exact commands
and how to wipe stale CMake cache.

### Upstreaming a routine to JAX

**Do not read `UPSTREAMING.md` unless the task is explicitly to port a
feature as a JAX PR.** It contains detailed file-by-file instructions for
submitting to the JAX monorepo and is not relevant to normal development.

---

## Non-obvious pitfalls

### Python fallback lowering — platform fallback, not grad/vmap

`_ormqr_lowering` in `_src/lax/linalg.py` is registered as the **default** MLIR
lowering (all platforms). The CPU and GPU FFI lowerings override it on those
platforms. It runs on platforms with no registered FFI target (e.g. TPU, or
when the `.so` is absent). It is **not** the vmap path — batching is handled
by `standard_linalg_primitive`'s built-in batching rule plus the C++ kernel's
batch loop. Do not remove it.

### Private JAX imports in `_src/lax/linalg.py`

The Python fallback lowering uses `from jax._src.lax import lax` (internal),
not `from jax import lax` (public). This is necessary to access private
helpers such as `lax._eye`. Keep these imports as-is; the public API does not
expose them.

### `CopyIfDiffBuffer` is mandatory in every C++ kernel

XLA is not guaranteed to alias the input buffer to the output buffer. Without
`CopyIfDiffBuffer(c, c_out)` before the LAPACK call, the kernel silently
writes to the wrong buffer when XLA does not alias. See the `Kernel` method
in `lapack_kernels.cc` for the required pattern.

### `operand_output_aliases` in `_linalg_ffi_lowering`

The dict passed to `_linalg_ffi_lowering(target_name, operand_output_aliases={2: 0})`
maps **output index → input index** and tells XLA it may reuse the input
buffer for the output. Getting the mapping wrong, or omitting it, causes
either double allocation (wasteful) or silent buffer corruption.

### Test references must be JAX-computed

Compare results against a reference built from `jax._src.lax.linalg.householder_product`
or `jax.scipy.linalg`, **not** `numpy`/`scipy`. This ensures the test
exercises the same dtype-promotion path as the implementation. Tolerances:
`1e-4` for f32/c64, `1e-10` for f64/c128.

### Benchmark artifacts are committed

`benchmarks/results/` contains the CSV and PNGs generated by the last run of
`benchmarks/bench_least_squares.py`. Do not regenerate them unless
benchmarking is the actual task.

### `docs/conf.py` stubs the C extension

`conf.py` pre-populates `sys.modules` with inert stubs for `jaxtra._jaxtra`
and `jaxtra._jaxtra_cuda` so Sphinx can import the package without a built
`.so`. Do not add logic to those stubs.
