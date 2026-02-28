// csrc/lapack_kernels.h — LAPACK ORMQR kernel declarations.
//
// Defines OrthogonalQrMultiply<dtype>, mirroring the struct of the same name
// in JAX PR #35104 (jaxlib/cpu/lapack_kernels.h).
//
// Key difference from PR: `fn` is set at Python import time by
// _jaxtra.initialize() (jaxtra_module.cc), which extracts the raw pointer from
// scipy.linalg.cython_lapack.__pyx_capi__ via nanobind capsule.  This avoids
// linking against OpenBLAS at build time and lets jaxtra work with the user's
// existing scipy installation.
#pragma once

#include <complex>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace jaxtra {

namespace ffi = ::xla::ffi;

// ---------------------------------------------------------------------------
// OrthogonalQrMultiply<dtype>
// ---------------------------------------------------------------------------
// Mirrors OrthogonalQrMultiply in JAX PR #35104.
//
// `fn` is the raw LAPACK function pointer (sormqr_/dormqr_/cunmqr_/zunmqr_),
// set at Python import time by _jaxtra.initialize().
template <ffi::DataType dtype>
struct OrthogonalQrMultiply {
  using ValueType = ffi::NativeType<dtype>;

  // Fortran LAPACK calling convention for all four variants.
  // For real dtypes: ValueType = float / double.
  // For complex dtypes: ValueType = std::complex<float/double>.
  using FnType = void(char* /*side*/, char* /*trans*/, int* /*m*/,
                      int* /*n*/, int* /*k*/, ValueType* /*a*/,
                      int* /*lda*/, ValueType* /*tau*/, ValueType* /*c*/,
                      int* /*ldc*/, ValueType* /*work*/, int* /*lwork*/,
                      int* /*info*/);

  // Function pointer; nullptr until initialize() is called.
  inline static FnType* fn = nullptr;

  // Apply Q to c_out in place (c_out is a copy of c on entry).
  static ffi::Error Kernel(ffi::Buffer<dtype> a, ffi::Buffer<dtype> tau,
                            ffi::Buffer<dtype> c, bool left, bool transpose,
                            ffi::ResultBuffer<dtype> c_out);

  // LAPACK workspace query.  Returns -1 on failure.
  static int64_t GetWorkspaceSize(char side, char trans, int m, int n, int k,
                                  int lda);
};

// Explicit instantiation declarations (definitions in lapack_kernels.cc).
extern template struct OrthogonalQrMultiply<ffi::DataType::F32>;
extern template struct OrthogonalQrMultiply<ffi::DataType::F64>;
extern template struct OrthogonalQrMultiply<ffi::DataType::C64>;
extern template struct OrthogonalQrMultiply<ffi::DataType::C128>;

}  // namespace jaxtra
