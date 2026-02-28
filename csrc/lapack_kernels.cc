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
// Explicit instantiations
// ---------------------------------------------------------------------------
template struct OrthogonalQrMultiply<ffi::DataType::F32>;
template struct OrthogonalQrMultiply<ffi::DataType::F64>;
template struct OrthogonalQrMultiply<ffi::DataType::C64>;
template struct OrthogonalQrMultiply<ffi::DataType::C128>;

}  // namespace jaxtra
