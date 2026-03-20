// nanobind module for jaxtra: registers XLA FFI LAPACK kernels.
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
using namespace jaxtra;

// ---------------------------------------------------------------------------
// AssignKernelFn — mirrors jaxlib's AssignKernelFn template.
// Sets the static fn pointer on a typed kernel struct.
// ---------------------------------------------------------------------------
template <typename Kernel>
void AssignKernelFn(void* fn) {
  Kernel::fn = reinterpret_cast<typename Kernel::FnType*>(fn);
}

// AssignKernelFnHe — assigns the hetrf function pointer (fn_he member).
// Used for complex types only; real types leave fn_he as nullptr.
template <typename Kernel>
void AssignKernelFnHe(void* fn) {
  Kernel::fn_he = reinterpret_cast<typename Kernel::FnType*>(fn);
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

// SYTRF — LDL decomposition, symmetric variant (sytrf).
// Two result buffers: a_out (factored matrix, same dtype) and ipiv (int32).
#define JAXTRA_CPU_DEFINE_SYTRF(name, dtype)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                       \
      name, LdlDecomposition<dtype>::KernelSy,                         \
      ffi::Ffi::Bind()                                                 \
          .Arg<ffi::Buffer<dtype>>()              /* a */              \
          .Attr<bool>("lower")                                         \
          .Ret<ffi::Buffer<dtype>>()              /* a_out */          \
          .Ret<ffi::Buffer<ffi::DataType::S32>>()) /* ipiv_out */

// HETRF — LDL decomposition, Hermitian variant (hetrf, complex only).
#define JAXTRA_CPU_DEFINE_HETRF(name, dtype)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                       \
      name, LdlDecomposition<dtype>::KernelHe,                         \
      ffi::Ffi::Bind()                                                 \
          .Arg<ffi::Buffer<dtype>>()              /* a */              \
          .Attr<bool>("lower")                                         \
          .Ret<ffi::Buffer<dtype>>()              /* a_out */          \
          .Ret<ffi::Buffer<ffi::DataType::S32>>()) /* ipiv_out */

// ---------------------------------------------------------------------------
// Handler macro for pentadiagonal solve (LAPACK gbsv, KL=KU=2).
// ---------------------------------------------------------------------------
#define JAXTRA_CPU_DEFINE_GBSV(name, dtype)                      \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                  \
      name, PentadiagonalSolve<dtype>::Kernel,                    \
      ffi::Ffi::Bind()                                            \
          .Arg<ffi::Buffer<dtype>>() /* ds */                     \
          .Arg<ffi::Buffer<dtype>>() /* dl */                     \
          .Arg<ffi::Buffer<dtype>>() /* d  */                     \
          .Arg<ffi::Buffer<dtype>>() /* du */                     \
          .Arg<ffi::Buffer<dtype>>() /* dw */                     \
          .Arg<ffi::Buffer<dtype>>() /* b  */                     \
          .Ret<ffi::Buffer<dtype>>()) /* x (b_out) */

// ---------------------------------------------------------------------------
// Handler macro for Hermitian pentadiagonal solve (LAPACK pbsv, KD=2).
// ---------------------------------------------------------------------------
#define JAXTRA_CPU_DEFINE_PBSV(name, dtype)                      \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                  \
      name, HermitianPentadiagonalSolve<dtype>::Kernel,            \
      ffi::Ffi::Bind()                                            \
          .Arg<ffi::Buffer<dtype>>() /* d  */                     \
          .Arg<ffi::Buffer<dtype>>() /* du */                     \
          .Arg<ffi::Buffer<dtype>>() /* dw */                     \
          .Arg<ffi::Buffer<dtype>>() /* b  */                     \
          .Ret<ffi::Buffer<dtype>>()) /* x (b_out) */

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

// hetrf — Hermitian variant, complex only (no shetr/dhetrf in LAPACK).
JAXTRA_CPU_DEFINE_HETRF(lapack_chetrf_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_HETRF(lapack_zhetrf_ffi, ffi::DataType::C128);

JAXTRA_CPU_DEFINE_GBSV(lapack_sgbsv_ffi, ffi::DataType::F32);
JAXTRA_CPU_DEFINE_GBSV(lapack_dgbsv_ffi, ffi::DataType::F64);
JAXTRA_CPU_DEFINE_GBSV(lapack_cgbsv_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_GBSV(lapack_zgbsv_ffi, ffi::DataType::C128);

JAXTRA_CPU_DEFINE_PBSV(lapack_spbsv_ffi, ffi::DataType::F32);
JAXTRA_CPU_DEFINE_PBSV(lapack_dpbsv_ffi, ffi::DataType::F64);
JAXTRA_CPU_DEFINE_PBSV(lapack_cpbsv_ffi, ffi::DataType::C64);
JAXTRA_CPU_DEFINE_PBSV(lapack_zpbsv_ffi, ffi::DataType::C128);

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

NB_MODULE(_jaxtra, m) {
  m.doc() = "jaxtra C extension: LAPACK ORMQR, SYTRF/HETRF, GBSV, PBSV via XLA FFI";

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
    // GBSV — pentadiagonal solve (banded LU)
    AssignKernelFn<PentadiagonalSolve<ffi::DataType::F32>>(lapack_ptr("sgbsv"));
    AssignKernelFn<PentadiagonalSolve<ffi::DataType::F64>>(lapack_ptr("dgbsv"));
    AssignKernelFn<PentadiagonalSolve<ffi::DataType::C64>>(lapack_ptr("cgbsv"));
    AssignKernelFn<PentadiagonalSolve<ffi::DataType::C128>>(lapack_ptr("zgbsv"));
    // PBSV — Hermitian pentadiagonal solve (banded Cholesky)
    AssignKernelFn<HermitianPentadiagonalSolve<ffi::DataType::F32>>(lapack_ptr("spbsv"));
    AssignKernelFn<HermitianPentadiagonalSolve<ffi::DataType::F64>>(lapack_ptr("dpbsv"));
    AssignKernelFn<HermitianPentadiagonalSolve<ffi::DataType::C64>>(lapack_ptr("cpbsv"));
    AssignKernelFn<HermitianPentadiagonalSolve<ffi::DataType::C128>>(lapack_ptr("zpbsv"));
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
    make_entry("lapack_chetrf_ffi", reinterpret_cast<void*>(lapack_chetrf_ffi));
    make_entry("lapack_zhetrf_ffi", reinterpret_cast<void*>(lapack_zhetrf_ffi));
    make_entry("lapack_sgbsv_ffi",  reinterpret_cast<void*>(lapack_sgbsv_ffi));
    make_entry("lapack_dgbsv_ffi",  reinterpret_cast<void*>(lapack_dgbsv_ffi));
    make_entry("lapack_cgbsv_ffi",  reinterpret_cast<void*>(lapack_cgbsv_ffi));
    make_entry("lapack_zgbsv_ffi",  reinterpret_cast<void*>(lapack_zgbsv_ffi));
    make_entry("lapack_spbsv_ffi",  reinterpret_cast<void*>(lapack_spbsv_ffi));
    make_entry("lapack_dpbsv_ffi",  reinterpret_cast<void*>(lapack_dpbsv_ffi));
    make_entry("lapack_cpbsv_ffi",  reinterpret_cast<void*>(lapack_cpbsv_ffi));
    make_entry("lapack_zpbsv_ffi",  reinterpret_cast<void*>(lapack_zpbsv_ffi));
    out["cpu"] = cpu_targets;
    return out;
  });
}
