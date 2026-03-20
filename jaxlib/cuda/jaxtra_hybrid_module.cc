// nanobind module for jaxtra hybrid GPU kernels: LAPACK hetrf running on
// CPU with explicit GPU↔CPU memory transfers.
//
// Mirrors jaxlib's _hybrid.so pattern:
//   - initialize() loads chetrf/zhetrf pointers from scipy.
//   - registrations() returns {platform: [(name, capsule, api_version)]}.
//
// These kernels are registered on the CUDA platform.  Each kernel receives GPU
// buffer pointers and handles the D→H / LAPACK / H→D transfer internally.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "jaxlib/gpu/hybrid_kernels_ffi.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = ::xla::ffi;

using namespace jax::JAX_GPU_NAMESPACE;

// AssignFn — sets the LAPACK function pointer on the typed hybrid kernel struct.
template <typename Kernel>
void AssignFn(void* fn) {
  Kernel::fn = reinterpret_cast<typename Kernel::FnType*>(fn);
}

NB_MODULE(_jaxtra_hybrid, m) {
  m.doc() = "jaxtra hybrid GPU extension: LAPACK hetrf on CPU, GPU-dispatched";

  // initialize() — loads LAPACK hetrf/hetrs pointers from scipy's Cython LAPACK
  // capsules, exactly mirroring jaxlib's GetLapackKernelsFromScipy().
  m.def("initialize", []() {
    nb::module_ cython_lapack =
        nb::module_::import_("scipy.linalg.cython_lapack");
    nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
    auto lapack_ptr = [&](const char* name) -> void* {
      return nb::cast<nb::capsule>(lapack_capi[name]).data();
    };
    AssignFn<HybridHetDecomp<ffi::DataType::C64>>(lapack_ptr("chetrf"));
    AssignFn<HybridHetDecomp<ffi::DataType::C128>>(lapack_ptr("zhetrf"));
  });

  // registrations() — returns {platform: [(name, capsule, api_version)]}
  // matching jaxlib.gpu_solver.registrations() format.
  m.def("registrations", []() {
    nb::dict out;
    nb::list gpu_targets;
    auto make_entry = [&](const char* name, void* sym) {
      gpu_targets.append(nb::make_tuple(
          name,
          nb::capsule(sym, "xla._CUSTOM_CALL_TARGET"),
          /*api_version=*/1));
    };
    make_entry("cuhybrid_chetrf_ffi",
               reinterpret_cast<void*>(cuhybrid_chetrf_ffi));
    make_entry("cuhybrid_zhetrf_ffi",
               reinterpret_cast<void*>(cuhybrid_zhetrf_ffi));
    out["CUDA"] = gpu_targets;
    return out;
  });
}
