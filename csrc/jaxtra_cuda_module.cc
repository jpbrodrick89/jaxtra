// nanobind module for jaxtra GPU (CUDA) support: registers the cuSolver ormqr
// XLA FFI handler so JAX can dispatch ormqr to the GPU.
//
// Mirrors jaxlib/gpu/solver.cc (Registrations / NB_MODULE) and the ormqr
// entry in jaxlib/gpu/gpu_kernels.cc (XLA_FFI_REGISTER_HANDLER), adapted for
// the non-Bazel build.
//
// Non-Bazel differences from PR #35104:
//   - Single module per feature (not linked into a monolithic jaxlib .so).
//   - CUDA headers found via CMake CUDAToolkit, not Bazel third_party/gpus.
//   - Abseil found via CMake find_package / FetchContent, not Bazel @com_google_absl.
//   - Include paths adjusted to resolve jaxlib/* under csrc/gpu/.
//   - XLA_FFI_REGISTER_HANDLER (static init) omitted; targets are registered
//     dynamically from Python via jax.ffi.register_ffi_target(), matching the
//     existing CPU registration pattern in jaxtra_module.cc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "jaxlib/gpu/solver_kernels_ffi.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;

// JAX_GPU_NAMESPACE is defined by vendor.h as "cuda" (CUDA) or "rocm" (HIP).
// JAX_GPU_PREFIX  is "cu" (CUDA) or "hip" (HIP).
using namespace jax::JAX_GPU_NAMESPACE;

NB_MODULE(_jaxtra_cuda, m) {
  m.doc() = "jaxtra CUDA extension: cuSolver ormqr via XLA FFI";

  // registrations() — returns {platform: [(name, capsule, api_version)]}
  // matching jaxlib.gpu_solver.registrations() format and the CPU equivalent
  // in jaxtra_module.cc.
  //
  // "CUDA" is the XLA platform string for CUDA; JAX maps it from the
  // target_name_prefix "cu" used in _ormqr_cpu_gpu_lowering.
  m.def("registrations", []() {
    nb::dict out;
    nb::list gpu_targets;
    auto make_entry = [&](const char* name, void* sym) {
      gpu_targets.append(nb::make_tuple(
          name,
          nb::capsule(sym, "xla._CUSTOM_CALL_TARGET"),
          /*api_version=*/1));
    };
    make_entry(JAX_GPU_PREFIX "solver_ormqr_ffi",
               reinterpret_cast<void*>(OrmqrFfi));
#ifdef JAX_GPU_CUDA
    make_entry(JAX_GPU_PREFIX "solver_sytrf_ffi",
               reinterpret_cast<void*>(SytrfFfi));
#endif  // JAX_GPU_CUDA
    out["CUDA"] = gpu_targets;
    return out;
  });
}
