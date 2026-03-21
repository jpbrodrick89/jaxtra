// jaxlib/gpu/hybrid_kernels_ffi.cc — LAPACK hetrf on CPU, GPU-dispatched.
//
// Each kernel:
//   1. Synchronises the GPU stream (ensures inputs are ready).
//   2. Copies device buffers to host.
//   3. Calls LAPACK on the host.
//   4. Copies results back to device (asynchronously on the stream).
//
// Compiles for both CUDA and ROCm via jaxlib/gpu/vendor.h.

#include "jaxlib/gpu/hybrid_kernels_ffi.h"

#include <algorithm>
#include <complex>
#include <vector>

#include "absl/status/status.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

// ---------------------------------------------------------------------------
// Macro for handler definitions.
// ---------------------------------------------------------------------------
#define JAXTRA_GPU_DEFINE_HYBRID_HETRF(name, dtype)                           \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                               \
      name, HybridHetDecomp<dtype>::Kernel,                                    \
      ffi::Ffi::Bind()                                                         \
          .Ctx<ffi::PlatformStream<gpuStream_t>>()                             \
          .Arg<ffi::Buffer<dtype>>()              /* a */                      \
          .Attr<bool>("lower")                                                  \
          .Ret<ffi::Buffer<dtype>>()              /* a_out */                  \
          .Ret<ffi::Buffer<ffi::DataType::S32>>()) /* ipiv_out */

// ---------------------------------------------------------------------------
// HybridHetDecomp::Kernel
// ---------------------------------------------------------------------------
template <ffi::DataType dtype>
ffi::Error HybridHetDecomp<dtype>::Kernel(
    gpuStream_t stream,
    ffi::Buffer<dtype> a, bool lower,
    ffi::ResultBuffer<dtype> a_out,
    ffi::ResultBuffer<ffi::DataType::S32> ipiv_out) {
  if (fn == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "HybridHetDecomp: LAPACK hetrf not initialized — "
                      "call _jaxtra_hybrid.initialize() first");
  }

  FFI_ASSIGN_OR_RETURN((auto [batch, n_rows, n_cols]),
                       SplitBatch2D(a.dimensions()));
  if (n_rows != n_cols) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "HybridHetDecomp: matrix must be square");
  }
  int n = static_cast<int>(n_rows);
  char uplo = lower ? 'L' : 'U';

  // Workspace query (lwork = -1).
  ValueType work_query = {};
  int lwork_q = -1, info = 0;
  fn(&uplo, &n, nullptr, &n, nullptr, &work_query, &lwork_q, &info);
  int lwork = info == 0 ? std::max(static_cast<int>(std::real(work_query)), 1)
                        : 1;

  const int64_t a_elems    = n_rows * n_cols;
  const int64_t ipiv_elems = n_rows;

  std::vector<ValueType> h_a(batch * a_elems);
  std::vector<int32_t>   h_ipiv(batch * ipiv_elems);
  std::vector<ValueType> h_work(lwork);

  // Fence: ensure pending GPU ops writing `a` are complete before D→H copy.
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(h_a.data(), a.typed_data(),
                     batch * a_elems * sizeof(ValueType),
                     gpuMemcpyDeviceToHost, stream));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  // Call LAPACK hetrf for each batch element.
  for (int64_t i = 0; i < batch; ++i) {
    fn(&uplo, &n,
       h_a.data()    + i * a_elems, &n,
       reinterpret_cast<int*>(h_ipiv.data() + i * ipiv_elems),
       h_work.data(), &lwork, &info);
    // info != 0 means the matrix is singular; mirror CPU behaviour and ignore.
  }

  // Copy results back to device asynchronously on the stream.
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(a_out->typed_data(), h_a.data(),
                     batch * a_elems * sizeof(ValueType),
                     gpuMemcpyHostToDevice, stream));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(ipiv_out->typed_data(), h_ipiv.data(),
                     batch * ipiv_elems * sizeof(int32_t),
                     gpuMemcpyHostToDevice, stream));

  // The host buffers must stay alive until the async copies finish.
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------
template struct HybridHetDecomp<ffi::DataType::C64>;
template struct HybridHetDecomp<ffi::DataType::C128>;

// ---------------------------------------------------------------------------
// Handler definitions
// ---------------------------------------------------------------------------
JAXTRA_GPU_DEFINE_HYBRID_HETRF(cuhybrid_chetrf_ffi, ffi::DataType::C64);
JAXTRA_GPU_DEFINE_HYBRID_HETRF(cuhybrid_zhetrf_ffi, ffi::DataType::C128);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
