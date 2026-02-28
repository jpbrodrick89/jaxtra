// LAPACK ORMQR kernel implementations for jaxtra.
//
// Layout contract:
//   The XLA FFI lowering rule specifies column-major (Fortran) layouts for
//   all 2-D matrix buffers.  XLA reorders data before calling these handlers,
//   so by the time execution reaches here every matrix is already in Fortran
//   (column-major) order — exactly what LAPACK expects.
//
// LAPACK DORMQR/SORMQR/CUNMQR/ZUNMQR signature:
//   ormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info)
//
//   SIDE  = 'L'  → Q @ C  (left:  A is m×k, C is m×n)
//   SIDE  = 'R'  → C @ Q  (right: A is n×k, C is m×n, n = Q size)
//   TRANS = 'N'  → use Q
//   TRANS = 'T'  → use Qᵀ (real only; complex uses 'C')
//   TRANS = 'C'  → use Qᴴ (complex only)
//
// All scalar integers are LP64 (int = 32 bit).
#include "lapack_kernels.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace jaxtra {

namespace {

// Copy n elements from src to dst.
template <typename T>
void CopyN(const T* src, T* dst, int64_t n) {
  std::memcpy(dst, src, n * sizeof(T));
}

// Core kernel: calls LAPACK ormqr for a single (possibly batched) problem.
// All matrix buffers arrive in Fortran (column-major) order.
// IsComplex is a compile-time constant so the dead branch for trans is
// eliminated by the compiler, matching the PR's `if constexpr` pattern.
template <typename T, auto LapackFn, bool IsComplex>
ffi::Error OrmqrKernel(const T* a_ptr, const T* tau_ptr, const T* c_ptr,
                       T* out_ptr,
                       ffi::Span<const int64_t> a_dims,
                       ffi::Span<const int64_t> c_dims,
                       bool left, bool transpose) {
  // Ranks: a is ≥2-D, c is ≥2-D, tau is ≥1-D.
  int na = static_cast<int>(a_dims.size());
  int nc = static_cast<int>(c_dims.size());

  // Matrix (last two) dimensions — in Fortran order these are rows × cols.
  int64_t a_rows = a_dims[na - 2];   // m: rows of Householder matrix
  int64_t k      = a_dims[na - 1];   // k: number of reflectors

  int64_t c_rows = c_dims[nc - 2];   // rows of C
  int64_t c_cols = c_dims[nc - 1];   // cols of C

  // Batch size (product of all but last 2 dims of C).
  int64_t batch = 1;
  for (int i = 0; i < nc - 2; ++i) batch *= c_dims[i];

  int64_t a_stride   = a_rows * k;
  int64_t c_stride   = c_rows * c_cols;
  int64_t tau_stride = k;

  char side  = left ? 'L' : 'R';
  // Real routines accept 'N'/'T'; complex routines accept 'N'/'C'.
  // The branch is resolved at compile time via the IsComplex template parameter.
  char trans;
  if constexpr (IsComplex) {
    trans = transpose ? 'C' : 'N';
  } else {
    trans = transpose ? 'T' : 'N';
  }

  // LAPACK integer arguments.
  int lm  = static_cast<int>(c_rows);  // rows of C
  int ln  = static_cast<int>(c_cols);  // cols of C
  int lk  = static_cast<int>(k);
  // LDA: leading dimension of A.  For left, A is a_rows×k; for right, A is c_cols×k.
  int lda = static_cast<int>(left ? a_rows : c_cols);
  int ldc = lm;
  int info = 0;

  // Workspace query.
  int lwork_query = -1;
  T work_size = T(0);
  LapackFn(&side, &trans, &lm, &ln, &lk,
           const_cast<T*>(a_ptr), &lda,
           const_cast<T*>(tau_ptr),
           const_cast<T*>(c_ptr),  // dummy C pointer; query doesn't touch it
           &ldc, &work_size, &lwork_query, &info);
  if (info != 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "LAPACK ormqr workspace query failed");
  }
  int lwork = static_cast<int>(std::real(work_size));
  if (lwork < 1) lwork = std::max({lm, ln, 1});
  std::vector<T> work(lwork);

  for (int64_t b = 0; b < batch; ++b) {
    const T* a_b   = a_ptr   + b * a_stride;
    const T* c_b   = c_ptr   + b * c_stride;
    T*       out_b = out_ptr + b * c_stride;

    // LAPACK operates in-place; copy C to output first.
    CopyN(c_b, out_b, c_stride);

    LapackFn(&side, &trans, &lm, &ln, &lk,
             const_cast<T*>(a_b), &lda,
             const_cast<T*>(tau_ptr) + b * tau_stride,
             out_b, &ldc,
             work.data(), &lwork, &info);

    if (info != 0) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        "LAPACK ormqr returned non-zero info");
    }
  }
  return ffi::Error::Success();
}

}  // namespace

// ---------------------------------------------------------------------------
// Float32
// ---------------------------------------------------------------------------
ffi::Error OrmqrF32(ffi::Buffer<ffi::DataType::F32> a,
                    ffi::Buffer<ffi::DataType::F32> tau,
                    ffi::Buffer<ffi::DataType::F32> c,
                    ffi::ResultBuffer<ffi::DataType::F32> c_out,
                    bool left, bool transpose) {
  return OrmqrKernel<float, sormqr_, /*IsComplex=*/false>(
      a.typed_data(), tau.typed_data(), c.typed_data(),
      c_out->typed_data(),
      a.dimensions(), c.dimensions(),
      left, transpose);
}

// ---------------------------------------------------------------------------
// Float64
// ---------------------------------------------------------------------------
ffi::Error OrmqrF64(ffi::Buffer<ffi::DataType::F64> a,
                    ffi::Buffer<ffi::DataType::F64> tau,
                    ffi::Buffer<ffi::DataType::F64> c,
                    ffi::ResultBuffer<ffi::DataType::F64> c_out,
                    bool left, bool transpose) {
  return OrmqrKernel<double, dormqr_, /*IsComplex=*/false>(
      a.typed_data(), tau.typed_data(), c.typed_data(),
      c_out->typed_data(),
      a.dimensions(), c.dimensions(),
      left, transpose);
}

// ---------------------------------------------------------------------------
// Complex64
// ---------------------------------------------------------------------------
ffi::Error OrmqrC64(ffi::Buffer<ffi::DataType::C64> a,
                    ffi::Buffer<ffi::DataType::C64> tau,
                    ffi::Buffer<ffi::DataType::C64> c,
                    ffi::ResultBuffer<ffi::DataType::C64> c_out,
                    bool left, bool transpose) {
  using CF = std::complex<float>;
  return OrmqrKernel<CF, cunmqr_, /*IsComplex=*/true>(
      reinterpret_cast<const CF*>(a.typed_data()),
      reinterpret_cast<const CF*>(tau.typed_data()),
      reinterpret_cast<const CF*>(c.typed_data()),
      reinterpret_cast<CF*>(c_out->typed_data()),
      a.dimensions(), c.dimensions(),
      left, transpose);
}

// ---------------------------------------------------------------------------
// Complex128
// ---------------------------------------------------------------------------
ffi::Error OrmqrC128(ffi::Buffer<ffi::DataType::C128> a,
                     ffi::Buffer<ffi::DataType::C128> tau,
                     ffi::Buffer<ffi::DataType::C128> c,
                     ffi::ResultBuffer<ffi::DataType::C128> c_out,
                     bool left, bool transpose) {
  using CD = std::complex<double>;
  return OrmqrKernel<CD, zunmqr_, /*IsComplex=*/true>(
      reinterpret_cast<const CD*>(a.typed_data()),
      reinterpret_cast<const CD*>(tau.typed_data()),
      reinterpret_cast<const CD*>(c.typed_data()),
      reinterpret_cast<CD*>(c_out->typed_data()),
      a.dimensions(), c.dimensions(),
      left, transpose);
}

}  // namespace jaxtra
