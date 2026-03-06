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
// Handler macros
// ---------------------------------------------------------------------------

// ORMQR — mirrors JAX_CPU_DEFINE_ORMQR in jaxlib upstream.
#define JAXTRA_CPU_DEFINE_ORMQR(name, dtype)                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                  \
      name, OrthogonalQrMultiply<dtype>::Kernel,                  \
      ffi::Ffi::Bind()                                            \
          .Arg<ffi::Buffer<dtype>>() /* a */                      \
          .Arg<ffi::Buffer<dtype>>() /* tau */                    \
          .Arg<ffi::Buffer<dtype>>() /* c */                      \
          .Attr<bool>("left")                                     \
          .Attr<bool>("transpose")                                \
          .Ret<ffi::Buffer<dtype>>()) /* c_out */

// SYTRF — LDL decomposition (symmetric / Hermitian indefinite).
// Two result buffers: a_out (factored matrix, same dtype) and ipiv (int32).
#define JAXTRA_CPU_DEFINE_SYTRF(name, dtype)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                       \
      name, LdlDecomposition<dtype>::Kernel,                           \
      ffi::Ffi::Bind()                                                 \
          .Arg<ffi::Buffer<dtype>>()              /* a */              \
          .Attr<bool>("lower")                                         \
          .Attr<bool>("hermitian")                                     \
          .Ret<ffi::Buffer<dtype>>()              /* a_out */          \
          .Ret<ffi::Buffer<ffi::DataType::S32>>()) /* ipiv_out */

// SYTRS — LDL solve (symmetric / Hermitian indefinite).
// Inputs: factors (packed sytrf output), ipiv (int32), b (rhs).
// One result buffer: x_out (solution, same dtype as factors/b).
#define JAXTRA_CPU_DEFINE_SYTRS(name, dtype)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                       \
      name, LdlSolve<dtype>::Kernel,                                   \
      ffi::Ffi::Bind()                                                 \
          .Arg<ffi::Buffer<dtype>>()               /* factors */       \
          .Arg<ffi::Buffer<ffi::DataType::S32>>()  /* ipiv */         \
          .Arg<ffi::Buffer<dtype>>()               /* b */             \
          .Attr<bool>("lower")                                         \
          .Attr<bool>("hermitian")                                     \
          .Ret<ffi::Buffer<dtype>>())              /* x_out */

// ---------------------------------------------------------------------------
// XLA FFI handler bindings (typed API, api_version=1).
// Names match JAX PR #35104.
// ---------------------------------------------------------------------------
JAXTRA_CPU_DEFINE_ORMQR(lapack_sormqr_ffi, ffi::DataType::F32);
JAXTRA_CPU_DEFINE_ORMQR(lapack_dormqr_ffi, ffi::DataType::F64);
JAXTRA_CPU_DEFINE_ORMQR(lapack_cunmqr_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_ORMQR(lapack_zunmqr_ffi, ffi::DataType::C128);

JAXTRA_CPU_DEFINE_SYTRF(lapack_ssytrf_ffi, ffi::DataType::F32);
JAXTRA_CPU_DEFINE_SYTRF(lapack_dsytrf_ffi, ffi::DataType::F64);
JAXTRA_CPU_DEFINE_SYTRF(lapack_csytrf_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_SYTRF(lapack_zsytrf_ffi, ffi::DataType::C128);

JAXTRA_CPU_DEFINE_SYTRS(lapack_ssytrs_ffi, ffi::DataType::F32);
JAXTRA_CPU_DEFINE_SYTRS(lapack_dsytrs_ffi, ffi::DataType::F64);
JAXTRA_CPU_DEFINE_SYTRS(lapack_csytrs_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_SYTRS(lapack_zsytrs_ffi, ffi::DataType::C128);

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

// AssignKernelFnHe — assigns the hetrf function pointer (fn_he member).
// Used for complex types only; real types leave fn_he as nullptr.
template <typename Kernel>
void AssignKernelFnHe(void* fn) {
  Kernel::fn_he = reinterpret_cast<typename Kernel::FnType*>(fn);
}

NB_MODULE(_jaxtra, m) {
  m.doc() = "jaxtra C extension: LAPACK ORMQR, SYTRF/HETRF, SYTRS/HETRS via XLA FFI";

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
    // ORMQR
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F32>>(lapack_ptr("sormqr"));
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::F64>>(lapack_ptr("dormqr"));
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C64>>(lapack_ptr("cunmqr"));
    AssignKernelFn<OrthogonalQrMultiply<ffi::DataType::C128>>(lapack_ptr("zunmqr"));
    // SYTRF — symmetric (real and complex symmetric)
    AssignKernelFn<LdlDecomposition<ffi::DataType::F32>>(lapack_ptr("ssytrf"));
    AssignKernelFn<LdlDecomposition<ffi::DataType::F64>>(lapack_ptr("dsytrf"));
    AssignKernelFn<LdlDecomposition<ffi::DataType::C64>>(lapack_ptr("csytrf"));
    AssignKernelFn<LdlDecomposition<ffi::DataType::C128>>(lapack_ptr("zsytrf"));
    // HETRF — Hermitian (complex only; real types leave fn_he as nullptr)
    AssignKernelFnHe<LdlDecomposition<ffi::DataType::C64>>(lapack_ptr("chetrf"));
    AssignKernelFnHe<LdlDecomposition<ffi::DataType::C128>>(lapack_ptr("zhetrf"));
    // SYTRS — symmetric solve (real and complex symmetric)
    AssignKernelFn<LdlSolve<ffi::DataType::F32>>(lapack_ptr("ssytrs"));
    AssignKernelFn<LdlSolve<ffi::DataType::F64>>(lapack_ptr("dsytrs"));
    AssignKernelFn<LdlSolve<ffi::DataType::C64>>(lapack_ptr("csytrs"));
    AssignKernelFn<LdlSolve<ffi::DataType::C128>>(lapack_ptr("zsytrs"));
    // HETRS — Hermitian solve (complex only; real types leave fn_he as nullptr)
    AssignKernelFnHe<LdlSolve<ffi::DataType::C64>>(lapack_ptr("chetrs"));
    AssignKernelFnHe<LdlSolve<ffi::DataType::C128>>(lapack_ptr("zhetrs"));
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
    make_entry("lapack_ssytrf_ffi", reinterpret_cast<void*>(lapack_ssytrf_ffi));
    make_entry("lapack_dsytrf_ffi", reinterpret_cast<void*>(lapack_dsytrf_ffi));
    make_entry("lapack_csytrf_ffi", reinterpret_cast<void*>(lapack_csytrf_ffi));
    make_entry("lapack_zsytrf_ffi", reinterpret_cast<void*>(lapack_zsytrf_ffi));
    make_entry("lapack_ssytrs_ffi", reinterpret_cast<void*>(lapack_ssytrs_ffi));
    make_entry("lapack_dsytrs_ffi", reinterpret_cast<void*>(lapack_dsytrs_ffi));
    make_entry("lapack_csytrs_ffi", reinterpret_cast<void*>(lapack_csytrs_ffi));
    make_entry("lapack_zsytrs_ffi", reinterpret_cast<void*>(lapack_zsytrs_ffi));
    out["cpu"] = cpu_targets;
    return out;
  });
}
