// pybind11 module for jaxtra: registers XLA FFI LAPACK ORMQR kernels.
//
// Pattern mirrors jaxlib.lapack: expose `registrations()` returning a dict
// of {platform: [(name, capsule, api_version)]} so the Python layer can call
// jax.ffi.register_ffi_target for each entry.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lapack_kernels.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// XLA FFI handler bindings (typed API, api_version=1)
// ---------------------------------------------------------------------------

// Bind each dtype variant via XLA_FFI_DEFINE_HANDLER_SYMBOL so that
// ffi::Ffi::BindTo returns the PyCapsule expected by register_custom_call_target.

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kOrmqrF32, jaxtra::OrmqrF32,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // a
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // tau
        .Arg<ffi::Buffer<ffi::DataType::F32>>()   // c
        .Ret<ffi::Buffer<ffi::DataType::F32>>()   // c_out
        .Attr<bool>("left")
        .Attr<bool>("transpose"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kOrmqrF64, jaxtra::OrmqrF64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Attr<bool>("left")
        .Attr<bool>("transpose"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kOrmqrC64, jaxtra::OrmqrC64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::C64>>()
        .Arg<ffi::Buffer<ffi::DataType::C64>>()
        .Arg<ffi::Buffer<ffi::DataType::C64>>()
        .Ret<ffi::Buffer<ffi::DataType::C64>>()
        .Attr<bool>("left")
        .Attr<bool>("transpose"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kOrmqrC128, jaxtra::OrmqrC128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Ret<ffi::Buffer<ffi::DataType::C128>>()
        .Attr<bool>("left")
        .Attr<bool>("transpose"));

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_jaxtra, m) {
  m.doc() = "jaxtra C extension: LAPACK ORMQR via XLA FFI";

  // Return registrations in the same format as jaxlib.lapack.registrations():
  //   dict[platform, list[tuple[name, capsule, api_version]]]
  // api_version=1 selects the typed XLA FFI (not the legacy custom-call API).
  m.def("registrations", []() {
    py::dict out;
    py::list cpu_targets;
    // Each entry: (name, PyCapsule, api_version)
    cpu_targets.append(py::make_tuple(
        "jaxtra_ormqr_f32",
        py::capsule(reinterpret_cast<void*>(kOrmqrF32), "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    cpu_targets.append(py::make_tuple(
        "jaxtra_ormqr_f64",
        py::capsule(reinterpret_cast<void*>(kOrmqrF64), "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    cpu_targets.append(py::make_tuple(
        "jaxtra_ormqr_c64",
        py::capsule(reinterpret_cast<void*>(kOrmqrC64), "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    cpu_targets.append(py::make_tuple(
        "jaxtra_ormqr_c128",
        py::capsule(reinterpret_cast<void*>(kOrmqrC128), "xla._CUSTOM_CALL_TARGET"),
        /*api_version=*/1));
    out["cpu"] = cpu_targets;
    return out;
  });
}
