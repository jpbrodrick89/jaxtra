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

// ---------------------------------------------------------------------------
// HermitianPentadiagonalSolve<dtype>
// ---------------------------------------------------------------------------
// Solves A X = B for X where A is a Hermitian (symmetric for real types)
// pentadiagonal matrix given by its upper triangle: d (main diagonal),
// du (offset +1), dw (offset +2).  Uses LAPACK pbsv (banded Cholesky, KD=2).
//
// Diagonal convention (upper triangle only):
//   d[i]  = A[i, i]
//   du[i] = A[i, i+1]   (unused for i = n-1)
//   dw[i] = A[i, i+2]   (unused for i >= n-2)
template <ffi::DataType dtype>
struct HermitianPentadiagonalSolve {
  using ValueType = ffi::NativeType<dtype>;

  // Fortran LAPACK calling convention for [sdcz]pbsv.
  using FnType = void(char* /*uplo*/, int* /*n*/, int* /*kd*/, int* /*nrhs*/,
                      ValueType* /*ab*/, int* /*ldab*/,
                      ValueType* /*b*/, int* /*ldb*/, int* /*info*/);

  // Function pointer; nullptr until initialize() is called.
  inline static FnType* fn = nullptr;

  // Solve A x = b for x, b_out is a copy of b on entry and holds x on exit.
  static ffi::Error Kernel(ffi::Buffer<dtype> d,  ffi::Buffer<dtype> du,
                            ffi::Buffer<dtype> dw, ffi::Buffer<dtype> b,
                            ffi::ResultBuffer<dtype> b_out);
};

// Explicit instantiation declarations (definitions in lapack_kernels.cc).
extern template struct HermitianPentadiagonalSolve<ffi::DataType::F32>;
extern template struct HermitianPentadiagonalSolve<ffi::DataType::F64>;
extern template struct HermitianPentadiagonalSolve<ffi::DataType::C64>;
extern template struct HermitianPentadiagonalSolve<ffi::DataType::C128>;

// ---------------------------------------------------------------------------
// LdlDecomposition<dtype>
// ---------------------------------------------------------------------------
// Symmetric/Hermitian indefinite factorization via LAPACK sytrf / hetrf.
//
// `fn`    — sytrf pointer (ssytrf/dsytrf for real; csytrf/zsytrf for complex
//            symmetric). Set at import time by _jaxtra.initialize().
// `fn_he` — hetrf pointer (chetrf/zhetrf for complex Hermitian only).
//            nullptr for real types.
//
// Both function pointers share the same Fortran calling convention.
template <ffi::DataType dtype>
struct LdlDecomposition {
  using ValueType = ffi::NativeType<dtype>;

  // Fortran LAPACK calling convention for sytrf / hetrf (identical signature).
  using FnType = void(char* /*uplo*/, int* /*n*/, ValueType* /*a*/,
                      int* /*lda*/, int* /*ipiv*/, ValueType* /*work*/,
                      int* /*lwork*/, int* /*info*/);

  // fn: sytrf variant; fn_he: hetrf variant (nullptr for real types).
  inline static FnType* fn = nullptr;
  inline static FnType* fn_he = nullptr;

  // Kernel: factors a (in-place → a_out), writes pivot indices to ipiv_out.
  static ffi::Error Kernel(ffi::Buffer<dtype> a, bool lower, bool hermitian,
                            ffi::ResultBuffer<dtype> a_out,
                            ffi::ResultBuffer<ffi::DataType::S32> ipiv_out);

  // LAPACK workspace query.  Returns -1 on failure.
  static int64_t GetWorkspaceSize(char uplo, int n, bool hermitian);
};

// Explicit instantiation declarations (definitions in lapack_kernels.cc).
extern template struct LdlDecomposition<ffi::DataType::F32>;
extern template struct LdlDecomposition<ffi::DataType::F64>;
extern template struct LdlDecomposition<ffi::DataType::C64>;
extern template struct LdlDecomposition<ffi::DataType::C128>;

// ---------------------------------------------------------------------------
// LdlSolve<dtype>
// ---------------------------------------------------------------------------
// Solves A*X = B using the factorization computed by LdlDecomposition via
// LAPACK ssytrs/dsytrs (real symmetric), csytrs/zsytrs (complex symmetric),
// or chetrs/zhetrs (complex Hermitian).
//
// `fn`    — sytrs pointer. Set at import time by _jaxtra.initialize().
// `fn_he` — hetrs pointer (complex Hermitian only; nullptr for real types).
//
// Unlike sytrf, sytrs has no workspace parameter.
template <ffi::DataType dtype>
struct LdlSolve {
  using ValueType = ffi::NativeType<dtype>;

  // Fortran LAPACK calling convention for sytrs / hetrs (no workspace).
  using FnType = void(char* /*uplo*/, int* /*n*/, int* /*nrhs*/,
                      ValueType* /*a*/, int* /*lda*/, int* /*ipiv*/,
                      ValueType* /*b*/, int* /*ldb*/, int* /*info*/);

  // fn: sytrs variant; fn_he: hetrs variant (nullptr for real types).
  inline static FnType* fn = nullptr;
  inline static FnType* fn_he = nullptr;

  // Kernel: reads factors and ipiv, solves b in-place → x_out.
  static ffi::Error Kernel(ffi::Buffer<dtype> factors,
                            ffi::Buffer<ffi::DataType::S32> ipiv,
                            ffi::Buffer<dtype> b, bool lower, bool hermitian,
                            ffi::ResultBuffer<dtype> x_out);
};

// Explicit instantiation declarations (definitions in lapack_kernels.cc).
extern template struct LdlSolve<ffi::DataType::F32>;
extern template struct LdlSolve<ffi::DataType::F64>;
extern template struct LdlSolve<ffi::DataType::C64>;
extern template struct LdlSolve<ffi::DataType::C128>;

}  // namespace jaxtra
