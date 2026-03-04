// csrc/lapack_kernels.cc — LAPACK ORMQR kernel implementations.
//
// Mirrors OrthogonalQrMultiply in JAX PR #35104 (jaxlib/cpu/lapack_kernels.cc).
// Key differences from PR are documented in lapack_kernels.h.
//
// Layout contract:
//   The XLA FFI lowering rule specifies column-major (Fortran) layouts for
//   all 2-D matrix buffers.  XLA reorders data before calling these handlers,
//   so by the time execution reaches here every matrix is already in Fortran
//   (column-major) order — exactly what LAPACK expects.
#include "lapack_kernels.h"
#include "lapack_utils.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

namespace jaxtra {

// ---------------------------------------------------------------------------
// GetWorkspaceSize
// ---------------------------------------------------------------------------
template <ffi::DataType dtype>
int64_t OrthogonalQrMultiply<dtype>::GetWorkspaceSize(char side, char trans,
                                                       int m, int n, int k,
                                                       int lda) {
  ValueType optimal_size = {};
  int c_leading_dim = m;
  int info = 0;
  int workspace_query = -1;
  // Pass nullptr for a/tau/c — LAPACK does not read them during a query.
  fn(&side, &trans, &m, &n, &k, /*a=*/nullptr, &lda, /*tau=*/nullptr,
     /*c=*/nullptr, &c_leading_dim, &optimal_size, &workspace_query, &info);
  return info == 0 ? static_cast<int64_t>(std::real(optimal_size)) : -1;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <ffi::DataType dtype>
ffi::Error OrthogonalQrMultiply<dtype>::Kernel(ffi::Buffer<dtype> a,
                                                ffi::Buffer<dtype> tau,
                                                ffi::Buffer<dtype> c,
                                                bool left, bool transpose,
                                                ffi::ResultBuffer<dtype> c_out) {
  // Unpack batch / matrix dimensions.
  auto dims_result = SplitBatch2D(c.dimensions());
  if (dims_result.has_error()) return std::move(dims_result.error());
  auto [batch_count, c_rows, c_cols] = *dims_result;

  auto a_dims_result = SplitBatch2D(a.dimensions());
  if (a_dims_result.has_error()) return std::move(a_dims_result.error());
  auto [a_batch, a_rows, a_cols] = *a_dims_result;

  // Narrow to LAPACK int, checking for overflow.
  FFI_ASSIGN_OR_RETURN(auto c_rows_v, MaybeCastNoOverflow<int>(c_rows));
  FFI_ASSIGN_OR_RETURN(auto c_cols_v, MaybeCastNoOverflow<int>(c_cols));
  FFI_ASSIGN_OR_RETURN(auto k_v,
                       MaybeCastNoOverflow<int>(tau.dimensions().back()));
  // LDA is the leading dimension of A in Fortran layout:
  //   left=true:  A is a_rows×k, LDA = a_rows
  //   left=false: A is c_cols×k, LDA = c_cols
  int lda_v = left ? static_cast<int>(a_rows) : c_cols_v;

  char side_v = left ? 'L' : 'R';
  char trans_v;
  // IsComplexType is a compile-time constexpr; the dead branch is eliminated.
  if constexpr (ffi::IsComplexType<dtype>()) {
    trans_v = transpose ? 'C' : 'N';
  } else {
    trans_v = transpose ? 'T' : 'N';
  }

  // Query the optimal workspace size once (all batch elements share shape).
  int64_t work_size = GetWorkspaceSize(side_v, trans_v, c_rows_v, c_cols_v,
                                       k_v, lda_v);
  if (work_size < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "LAPACK ormqr workspace query failed");
  }
  auto work_size_v = static_cast<int>(std::max(work_size, int64_t{1}));
  std::vector<ValueType> work(work_size_v);

  // Copy c → c_out only if XLA gave us separate buffers.
  // (When operand_output_aliases is set in the lowering rule, XLA may alias
  // input and output to the same buffer to avoid the copy.)
  CopyIfDiffBuffer(c, c_out);

  auto* a_data   = a.typed_data();
  auto* tau_data = tau.typed_data();
  auto* out_data = c_out->typed_data();

  int c_leading_dim = c_rows_v;
  int info = 0;  // ignored; matches jaxlib's behaviour

  const int64_t c_step   = c_rows * c_cols;
  const int64_t a_step   = a_rows * a_cols;
  const int64_t tau_step = tau.dimensions().back();

  for (int64_t i = 0; i < batch_count; ++i) {
    fn(&side_v, &trans_v, &c_rows_v, &c_cols_v, &k_v,
       const_cast<ValueType*>(a_data), &lda_v,
       const_cast<ValueType*>(tau_data),
       out_data, &c_leading_dim, work.data(), &work_size_v, &info);
    a_data   += a_step;
    tau_data += tau_step;
    out_data += c_step;
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Explicit instantiations — OrthogonalQrMultiply
// ---------------------------------------------------------------------------
template struct OrthogonalQrMultiply<ffi::DataType::F32>;
template struct OrthogonalQrMultiply<ffi::DataType::F64>;
template struct OrthogonalQrMultiply<ffi::DataType::C64>;
template struct OrthogonalQrMultiply<ffi::DataType::C128>;

// ===========================================================================
// PentadiagonalSolve — LAPACK gbsv (banded LU, KL=KU=2)
// ===========================================================================

template <ffi::DataType dtype>
ffi::Error PentadiagonalSolve<dtype>::Kernel(
    ffi::Buffer<dtype> ds, ffi::Buffer<dtype> dl, ffi::Buffer<dtype> d,
    ffi::Buffer<dtype> du, ffi::Buffer<dtype> dw, ffi::Buffer<dtype> b,
    ffi::ResultBuffer<dtype> b_out) {
  // Unpack batch / vector dimensions from diagonals (rank-1 per batch).
  auto d_dims_result = SplitBatch1D(d.dimensions());
  if (d_dims_result.has_error()) return std::move(d_dims_result.error());
  auto [batch_count, n] = *d_dims_result;

  // b has shape (..., n, nrhs) — use SplitBatch2D to extract nrhs.
  auto b_dims_result = SplitBatch2D(b.dimensions());
  if (b_dims_result.has_error()) return std::move(b_dims_result.error());
  auto [b_batch, b_rows, nrhs] = *b_dims_result;

  FFI_ASSIGN_OR_RETURN(auto n_v, MaybeCastNoOverflow<int>(n));
  FFI_ASSIGN_OR_RETURN(auto nrhs_v, MaybeCastNoOverflow<int>(nrhs));

  // LAPACK band storage parameters for pentadiagonal (KL=KU=2).
  const int kl = 2, ku = 2;
  // LDAB = 2*KL + KU + 1 = 7 (KL extra rows for LU pivoting + band width).
  const int ldab = 2 * kl + ku + 1;  // = 7
  int ldab_v = ldab;
  int ldb_v = n_v;

  // Allocate band storage matrix (column-major) and pivot array per batch.
  std::vector<ValueType> AB(static_cast<std::size_t>(ldab) * n_v);
  std::vector<int> ipiv(static_cast<std::size_t>(n_v));

  // Copy b -> b_out only when XLA allocated separate buffers.
  CopyIfDiffBuffer(b, b_out);

  auto* ds_data = ds.typed_data();
  auto* dl_data = dl.typed_data();
  auto* d_data  = d.typed_data();
  auto* du_data = du.typed_data();
  auto* dw_data = dw.typed_data();
  auto* x_data  = b_out->typed_data();  // solution overwrites b_out in place

  for (int64_t batch = 0; batch < batch_count; ++batch) {
    // Pack five diagonals into LAPACK band storage (column-major, ldab x n).
    //
    // LAPACK band storage: AB[ku + i - j, j] = A[i, j]
    // With KU=2: row index = 2 + i - j in band storage.
    //   Row 0-1: extra rows for LU (KL=2 extra pivot rows, left zero).
    //   Row 2:   upper super-diagonal 2 -- AB[2, j] = A[j-2, j] = dw[j-2]
    //   Row 3:   upper super-diagonal 1 -- AB[3, j] = A[j-1, j] = du[j-1]
    //   Row 4:   main diagonal          -- AB[4, j] = A[j,   j] = d[j]
    //   Row 5:   lower sub-diagonal 1  -- AB[5, j] = A[j+1, j] = dl[j+1]
    //   Row 6:   lower sub-diagonal 2  -- AB[6, j] = A[j+2, j] = ds[j+2]
    for (int j = 0; j < n_v; ++j) {
      int base = j * ldab;
      AB[base + 0] = ValueType{0};
      AB[base + 1] = ValueType{0};
      AB[base + 2] = (j >= 2)       ? dw_data[j - 2] : ValueType{0};
      AB[base + 3] = (j >= 1)       ? du_data[j - 1] : ValueType{0};
      AB[base + 4] = d_data[j];
      AB[base + 5] = (j <= n_v - 2) ? dl_data[j + 1] : ValueType{0};
      AB[base + 6] = (j <= n_v - 3) ? ds_data[j + 2] : ValueType{0};
    }

    int info = 0;
    fn(&n_v, const_cast<int*>(&kl), const_cast<int*>(&ku),
       &nrhs_v, AB.data(), &ldab_v, ipiv.data(),
       x_data, &ldb_v, &info);
    // info > 0 means singular; follow jaxlib's convention of not raising.

    ds_data += n;
    dl_data += n;
    d_data  += n;
    du_data += n;
    dw_data += n;
    x_data  += n * nrhs;
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Explicit instantiations — PentadiagonalSolve
// ---------------------------------------------------------------------------
template struct PentadiagonalSolve<ffi::DataType::F32>;
template struct PentadiagonalSolve<ffi::DataType::F64>;
template struct PentadiagonalSolve<ffi::DataType::C64>;
template struct PentadiagonalSolve<ffi::DataType::C128>;

// ===========================================================================
// HermitianPentadiagonalSolve — LAPACK pbsv (banded Cholesky, KD=2)
// ===========================================================================

template <ffi::DataType dtype>
ffi::Error HermitianPentadiagonalSolve<dtype>::Kernel(
    ffi::Buffer<dtype> d, ffi::Buffer<dtype> du, ffi::Buffer<dtype> dw,
    ffi::Buffer<dtype> b, ffi::ResultBuffer<dtype> b_out) {
  // Unpack batch / vector dimensions from main diagonal (rank-1 per batch).
  auto d_dims_result = SplitBatch1D(d.dimensions());
  if (d_dims_result.has_error()) return std::move(d_dims_result.error());
  auto [batch_count, n] = *d_dims_result;

  // b has shape (..., n, nrhs) — use SplitBatch2D to extract nrhs.
  auto b_dims_result = SplitBatch2D(b.dimensions());
  if (b_dims_result.has_error()) return std::move(b_dims_result.error());
  auto [b_batch, b_rows, nrhs] = *b_dims_result;

  FFI_ASSIGN_OR_RETURN(auto n_v, MaybeCastNoOverflow<int>(n));
  FFI_ASSIGN_OR_RETURN(auto nrhs_v, MaybeCastNoOverflow<int>(nrhs));

  // LAPACK band storage parameters for Hermitian pentadiagonal (KD=2).
  const int kd = 2;
  // LDAB = KD + 1 = 3 (no extra rows needed for Cholesky, unlike LU).
  const int ldab = kd + 1;  // = 3
  int ldab_v = ldab;
  int ldb_v = n_v;
  char uplo = 'U';

  // Allocate band storage matrix (column-major) per batch.
  std::vector<ValueType> AB(static_cast<std::size_t>(ldab) * n_v);

  // Copy b -> b_out only when XLA allocated separate buffers.
  CopyIfDiffBuffer(b, b_out);

  auto* d_data  = d.typed_data();
  auto* du_data = du.typed_data();
  auto* dw_data = dw.typed_data();
  auto* x_data  = b_out->typed_data();  // solution overwrites b_out in place

  for (int64_t batch = 0; batch < batch_count; ++batch) {
    // Pack upper triangle into LAPACK band storage (column-major, ldab x n).
    //
    // LAPACK upper band storage (UPLO='U'): AB[kd + i - j, j] = A[i, j]
    // With KD=2:
    //   Row 0: second super-diagonal -- AB[0, j] = A[j-2, j] = dw[j-2]
    //   Row 1: first super-diagonal  -- AB[1, j] = A[j-1, j] = du[j-1]
    //   Row 2: main diagonal         -- AB[2, j] = A[j,   j] = d[j]
    for (int j = 0; j < n_v; ++j) {
      int base = j * ldab;
      AB[base + 0] = (j >= 2) ? dw_data[j - 2] : ValueType{0};
      AB[base + 1] = (j >= 1) ? du_data[j - 1] : ValueType{0};
      AB[base + 2] = d_data[j];
    }

    int info = 0;
    fn(&uplo, &n_v, const_cast<int*>(&kd), &nrhs_v,
       AB.data(), &ldab_v, x_data, &ldb_v, &info);
    // info > 0 means not positive definite; follow jaxlib's convention.

    d_data  += n;
    du_data += n;
    dw_data += n;
    x_data  += n * nrhs;
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Explicit instantiations — HermitianPentadiagonalSolve
// ---------------------------------------------------------------------------
template struct HermitianPentadiagonalSolve<ffi::DataType::F32>;
template struct HermitianPentadiagonalSolve<ffi::DataType::F64>;
template struct HermitianPentadiagonalSolve<ffi::DataType::C64>;
template struct HermitianPentadiagonalSolve<ffi::DataType::C128>;

}  // namespace jaxtra
