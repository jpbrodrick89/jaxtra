// nanobind module for jaxtra: registers XLA FFI LAPACK ORMQR kernels.
//
// Uses nanobind (same as jaxlib) so that initialize() can extract raw LAPACK
// function pointers from scipy's Cython capsules via nb::capsule::data(),
// exactly mirroring jaxlib's GetLapackKernelsFromScipy().
//
// Handler names match JAX PR #35104 exactly (lapack_sormqr_ffi etc.) so that
// if the PR is merged, jaxtra users can switch without changing call sites.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "lapack_kernels.h"
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;
using namespace jaxtra;

// ---------------------------------------------------------------------------
// AssignKernelFn — mirrors jaxlib's AssignKernelFn template.
// Sets the static fn pointer on a typed kernel struct.
// ---------------------------------------------------------------------------
template <typename Kernel>
void AssignKernelFn(void* fn) {
  Kernel::fn = reinterpret_cast<typename Kernel::FnType*>(fn);
}

// ---------------------------------------------------------------------------
// Handler macro — mirrors JAX_CPU_DEFINE_ORMQR from the upstream PR.
// ---------------------------------------------------------------------------
#define JAXTRA_DEFINE_ORMQR(name, dtype)                         \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                  \
      name, OrthogonalQrMultiply<dtype>::Kernel,                  \
      ffi::Ffi::Bind()                                            \
          .Arg<ffi::Buffer<dtype>>() /* a */                      \
          .Arg<ffi::Buffer<dtype>>() /* tau */                    \
          .Arg<ffi::Buffer<dtype>>() /* c */                      \
          .Attr<bool>("left")                                     \
          .Attr<bool>("transpose")                                \
          .Ret<ffi::Buffer<dtype>>()) /* c_out */

// ---------------------------------------------------------------------------
// XLA FFI handler bindings (typed API, api_version=1).
// Names match JAX PR #35104.
// ---------------------------------------------------------------------------
JAXTRA_DEFINE_ORMQR(lapack_sormqr_ffi, ffi::DataType::F32);
JAXTRA_DEFINE_ORMQR(lapack_dormqr_ffi, ffi::DataType::F64);
JAXTRA_DEFINE_ORMQR(lapack_cunmqr_ffi, ffi::DataType::C64);
JAXTRA_DEFINE_ORMQR(lapack_zunmqr_ffi, ffi::DataType::C128);

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

NB_MODULE(_jaxtra, m) {
  m.doc() = "jaxtra C extension: LAPACK ORMQR via XLA FFI";

  // initialize() — mirrors jaxlib's GetLapackKernelsFromScipy().
  // Imports scipy.linalg.cython_lapack, extracts raw function pointers from
  // its __pyx_capi__ dict via nb::capsule::data() (no name check needed),
  // and assigns them to the typed kernel structs.
  m.def("initialize", []() {
    nb::module_ cython_lapack =
        nb::module_::import_("scipy.linalg.cython_lapack");
    nb::dict lapack_capi = cython_lapack.attr("__pyx_capi__");
    auto lapack_ptr = [&](const char* name) -> void* {
      return nb::cast<nb::capsule>(lapack_capi[name]).data();
    };
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32>>(lapack_ptr("sormqr"));
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64>>(lapack_ptr("dormqr"));
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64>>(lapack_ptr("cunmqr"));
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128>>(lapack_ptr("zunmqr"));
  });

  // registrations() — returns {platform: [(name, capsule, api_version)]}
  // matching jaxlib.lapack.registrations() format.
  m.def("registrations", []() {
    nb::dict out;
    nb::list cpu_targets;
    auto make_entry = [&](const char* name, void* sym) {
      cpu_targets.append(nb::make_tuple(
          name,
          nb::capsule(sym, "xla._CUSTOM_CALL_TARGET"),
          /*api_version=*/1));
    };
    make_entry("lapack_sormqr_ffi", reinterpret_cast<void*>(lapack_sormqr_ffi));
    make_entry("lapack_dormqr_ffi", reinterpret_cast<void*>(lapack_dormqr_ffi));
    make_entry("lapack_cunmqr_ffi", reinterpret_cast<void*>(lapack_cunmqr_ffi));
    make_entry("lapack_zunmqr_ffi", reinterpret_cast<void*>(lapack_zunmqr_ffi));
    out["cpu"] = cpu_targets;
    return out;
  });
}
