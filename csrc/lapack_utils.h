// csrc/lapack_utils.h — small utilities mirroring jaxlib internal helpers.
//
// jaxlib's internal headers (cpu_kernels.h, lapack_kernels.h) provide these
// utilities but are not part of the public XLA FFI API.  We reimplement them
// here using only public types from <xla/ffi/api/ffi.h>.
//
// Provides:
//   SplitBatch2D    — splits an N-D span into (batch, rows, cols)
//   MaybeCastNoOverflow<To>  — checked integer narrowing cast
//   CopyIfDiffBuffer         — in-place LAPACK copy guard
//   FFI_ASSIGN_OR_RETURN     — error-propagation macro (non-structured-binding)
#pragma once

#include <cstdint>
#include <limits>
#include <tuple>

#include "xla/ffi/api/ffi.h"

namespace xla::ffi {
// Bring Unexpected into scope so callers can write ffi::Unexpected(...)
using ::xla::ffi::Unexpected;
}  // namespace xla::ffi

namespace jaxtra {

namespace ffi = ::xla::ffi;

// ---------------------------------------------------------------------------
// SplitBatch2D
// ---------------------------------------------------------------------------
// Returns (batch_count, rows, cols) from an N-D dimension span.
// Mirrors jaxlib's SplitBatch2D helper.
using Dims3 = std::tuple<int64_t, int64_t, int64_t>;

inline ffi::ErrorOr<Dims3> SplitBatch2D(ffi::Span<const int64_t> dims) {
  if (dims.size() < 2) {
    return ffi::Unexpected(
        ffi::Error(ffi::ErrorCode::kInvalidArgument, "expected >= 2-D buffer"));
  }
  int64_t batch = 1;
  for (std::size_t i = 0; i + 2 < dims.size(); ++i) batch *= dims[i];
  return std::make_tuple(batch, dims[dims.size() - 2], dims[dims.size() - 1]);
}

// ---------------------------------------------------------------------------
// MaybeCastNoOverflow<To>
// ---------------------------------------------------------------------------
// Narrows int64_t to To; returns an error if the value would overflow.
// Mirrors jaxlib's MaybeCastNoOverflow<lapack_int>.
template <typename To>
inline ffi::ErrorOr<To> MaybeCastNoOverflow(int64_t v) {
  if (v > static_cast<int64_t>(std::numeric_limits<To>::max())) {
    return ffi::Unexpected(ffi::Error(ffi::ErrorCode::kInvalidArgument,
                                      "dimension overflows lapack_int"));
  }
  return static_cast<To>(v);
}

// ---------------------------------------------------------------------------
// CopyIfDiffBuffer
// ---------------------------------------------------------------------------
// Copies src into dst only when XLA has allocated distinct buffers.
// When operand_output_aliases is set, XLA may alias input and output to the
// same buffer; LAPACK requires the in/out array to be c_out, so we only copy
// when needed.  Mirrors jaxlib's CopyIfDiffBuffer helper.
template <ffi::DataType dtype>
inline void CopyIfDiffBuffer(ffi::Buffer<dtype> src,
                              ffi::ResultBuffer<dtype>& dst) {
  if (src.typed_data() != dst->typed_data()) {
    std::copy_n(src.typed_data(), src.element_count(), dst->typed_data());
  }
}

}  // namespace jaxtra

// ---------------------------------------------------------------------------
// FFI_ASSIGN_OR_RETURN
// ---------------------------------------------------------------------------
// Assigns the value from an ffi::ErrorOr<T> expression, or propagates the
// error by returning early.  Use for simple (non-structured-binding) cases.
// Structured binding form: call SplitBatch2D explicitly, then use auto [].
//
// Usage:
//   FFI_ASSIGN_OR_RETURN(auto c_rows_v, MaybeCastNoOverflow<int>(c_rows));
//
// The unique temp name is derived from __LINE__ (requires two-level expansion
// so the preprocessor fully expands __LINE__ before token-pasting).
#define FFI_CONCAT_(a, b) a##b
#define FFI_CONCAT(a, b) FFI_CONCAT_(a, b)
#define FFI_ASSIGN_OR_RETURN(var, expr)                            \
  auto FFI_CONCAT(_ffi_result_, __LINE__) = (expr);               \
  if (FFI_CONCAT(_ffi_result_, __LINE__).has_error())             \
    return std::move(FFI_CONCAT(_ffi_result_, __LINE__).error()); \
  var = *std::move(FFI_CONCAT(_ffi_result_, __LINE__))
