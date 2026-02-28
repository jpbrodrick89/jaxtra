// pybind11 module for jaxtra: registers XLA FFI LAPACK ORMQR kernels.
//
// Pattern mirrors jaxlib.lapack: expose `registrations()` returning a dict
// of {platform: [(name, capsule, api_version)]} so the Python layer can call
// jax.ffi.register_ffi_target for each entry.
//
// Handler names match JAX PR #35104 exactly (lapack_sormqr_ffi etc.) so that
// if the PR is merged, jaxtra users can switch without changing call sites.
//
// set_lapack_fn_ptrs() is called at import time by jaxtra._core using ctypes
// to extract raw pointers from scipy.linalg.cython_lapack.__pyx_capi__.  This
// avoids linking against OpenBLAS at wheel-build time.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>

#include "lapack_kernels.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;
using namespace jaxtra;

// ---------------------------------------------------------------------------
// Handler macro — mirrors JAX_CPU_DEFINE_ORMQR from the upstream PR.
// Binds a typed FFI handler for a single dtype, reducing the four-times
// repetition of the Ffi::Bind() chain to a single macro call.
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
// Names match JAX PR #35104 so that downstream code can switch seamlessly
// once the PR is merged into jaxlib proper.
// ---------------------------------------------------------------------------
JAXTRA_DEFINE_ORMQR(lapack_sormqr_ffi, ffi::DataType::F32);
JAXTRA_DEFINE_ORMQR(lapack_dormqr_ffi, ffi::DataType::F64);
JAXTRA_DEFINE_ORMQR(lapack_cunmqr_ffi, ffi::DataType::C64);
JAXTRA_DEFINE_ORMQR(lapack_zunmqr_ffi, ffi::DataType::C128);

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_jaxtra, m) {
  m.doc() = "jaxtra C extension: LAPACK ORMQR via XLA FFI";

  // set_lapack_fn_ptrs(sormqr, dormqr, cunmqr, zunmqr)
  // Called from jaxtra._core._register_targets() via ctypes, passing raw
  // function pointers extracted from scipy.linalg.cython_lapack.__pyx_capi__.
  // Mirrors jaxlib's initialize() mechanism: no LAPACK link at build time.
  m.def("set_lapack_fn_ptrs",
        [](uintptr_t sormqr_ptr, uintptr_t dormqr_ptr,
           uintptr_t cunmqr_ptr, uintptr_t zunmqr_ptr) {
          OrthogonalQrMultiply<ffi::DataType::F32>::fn =
              reinterpret_cast<OrthogonalQrMultiply<ffi::DataType::F32>::FnType>(sormqr_ptr);
          OrthogonalQrMultiply<ffi::DataType::F64>::fn =
              reinterpret_cast<OrthogonalQrMultiply<ffi::DataType::F64>::FnType>(dormqr_ptr);
          OrthogonalQrMultiply<ffi::DataType::C64>::fn =
              reinterpret_cast<OrthogonalQrMultiply<ffi::DataType::C64>::FnType>(cunmqr_ptr);
          OrthogonalQrMultiply<ffi::DataType::C128>::fn =
              reinterpret_cast<OrthogonalQrMultiply<ffi::DataType::C128>::FnType>(zunmqr_ptr);
        });

  // Return registrations in the same format as jaxlib.lapack.registrations():
  //   dict[platform, list[tuple[name, capsule, api_version]]]
  // api_version=1 selects the typed XLA FFI (not the legacy custom-call API).
  m.def("registrations", []() {
    py::dict out;
    py::list cpu_targets;
    // Each entry: (name, PyCapsule, api_version)
    cpu_targets.append(py::make_tuple(
        "lapack_sormqr_ffi",
        py::capsule(reinterpret_cast<void*>(lapack_sormqr_ffi),
                    "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    cpu_targets.append(py::make_tuple(
        "lapack_dormqr_ffi",
        py::capsule(reinterpret_cast<void*>(lapack_dormqr_ffi),
                    "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    cpu_targets.append(py::make_tuple(
        "lapack_cunmqr_ffi",
        py::capsule(reinterpret_cast<void*>(lapack_cunmqr_ffi),
                    "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    cpu_targets.append(py::make_tuple(
        "lapack_zunmqr_ffi",
        py::capsule(reinterpret_cast<void*>(lapack_zunmqr_ffi),
                    "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    out["cpu"] = cpu_targets;
    return out;
  });
}
