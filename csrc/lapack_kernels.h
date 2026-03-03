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

// ---------------------------------------------------------------------------
// PentadiagonalSolve<dtype>
// ---------------------------------------------------------------------------
// Solves A X = B for X where A is a pentadiagonal matrix given by five
// diagonals: ds (offset -2), dl (offset -1), d (main), du (offset +1),
// dw (offset +2).  Uses LAPACK gbsv (banded LU).
//
// Diagonal convention (matching cuSparse gpsvInterleavedBatch):
//   ds[i] = A[i, i-2]   (unused for i < 2)
//   dl[i] = A[i, i-1]   (unused for i < 1)
//   d[i]  = A[i, i]
//   du[i] = A[i, i+1]   (unused for i = n-1)
//   dw[i] = A[i, i+2]   (unused for i >= n-2)
template <ffi::DataType dtype>
struct PentadiagonalSolve {
  using ValueType = ffi::NativeType<dtype>;

  // Fortran LAPACK calling convention for [sdcz]gbsv.
  using FnType = void(int* /*n*/, int* /*kl*/, int* /*ku*/, int* /*nrhs*/,
                      ValueType* /*ab*/, int* /*ldab*/, int* /*ipiv*/,
                      ValueType* /*b*/, int* /*ldb*/, int* /*info*/);

  // Function pointer; nullptr until initialize() is called.
  inline static FnType* fn = nullptr;

  // Solve A x = b for x, b_out is a copy of b on entry and holds x on exit.
  static ffi::Error Kernel(ffi::Buffer<dtype> ds, ffi::Buffer<dtype> dl,
                            ffi::Buffer<dtype> d,  ffi::Buffer<dtype> du,
                            ffi::Buffer<dtype> dw, ffi::Buffer<dtype> b,
                            ffi::ResultBuffer<dtype> b_out);
};

// Explicit instantiation declarations (definitions in lapack_kernels.cc).
extern template struct PentadiagonalSolve<ffi::DataType::F32>;
extern template struct PentadiagonalSolve<ffi::DataType::F64>;
extern template struct PentadiagonalSolve<ffi::DataType::C64>;
extern template struct PentadiagonalSolve<ffi::DataType::C128>;

}  // namespace jaxtra
