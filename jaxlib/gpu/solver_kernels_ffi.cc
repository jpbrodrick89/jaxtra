/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Verbatim from JAX PR #35104: jaxlib/gpu/solver_kernels_ffi.cc (ormqr section).
// Only the OrmqrImpl / OrmqrDispatch / XLA_FFI_DEFINE_HANDLER_SYMBOL block is
// included; all other handlers are provided by the installed jaxlib GPU
// shared library.

#include "jaxlib/gpu/solver_kernels_ffi.h"

#include <cstdint>
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "jaxlib/gpu/solver_interface.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

// Real ormqr (Sormqr/Dormqr) accepts CUBLAS_OP_T for transpose;
// complex unmqr (Cunmqr/Zunmqr) requires CUBLAS_OP_C (conjugate transpose).
template <typename T>
gpublasOperation_t TransposeOp() {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return GPUBLAS_OP_T;
  } else {
    return GPUBLAS_OP_C;
  }
}

#define SOLVER_DISPATCH_IMPL(impl, ...)           \
  switch (dataType) {                             \
    case ffi::F32:                                \
      return impl<float>(__VA_ARGS__);            \
    case ffi::F64:                                \
      return impl<double>(__VA_ARGS__);           \
    case ffi::C64:                                \
      return impl<gpuComplex>(__VA_ARGS__);       \
    case ffi::C128:                               \
      return impl<gpuDoubleComplex>(__VA_ARGS__); \
    default:                                      \
      break;                                      \
  }

// Householder multiply: ormqr/unmqr

template <typename T>
ffi::Error OrmqrImpl(int64_t batch, int64_t c_rows, int64_t c_cols,
                     int64_t k, int64_t a_rows, int64_t a_cols,
                     bool left, bool transpose,
                     gpuStream_t stream, ffi::ScratchAllocator& scratch,
                     ffi::AnyBuffer a, ffi::AnyBuffer tau,
                     ffi::AnyBuffer c, ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(auto m, MaybeCastNoOverflow<int>(c_rows));
  FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(c_cols));
  FFI_ASSIGN_OR_RETURN(auto k_v, MaybeCastNoOverflow<int>(k));
  FFI_ASSIGN_OR_RETURN(auto lda, MaybeCastNoOverflow<int>(a_rows));

  gpublasSideMode_t side = left ? GPUBLAS_SIDE_LEFT : GPUBLAS_SIDE_RIGHT;
  gpublasOperation_t trans = transpose ? TransposeOp<T>() : GPUBLAS_OP_N;

  FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
  FFI_ASSIGN_OR_RETURN(int lwork,
                       solver::OrmqrBufferSize<T>(handle.get(), side, trans,
                                                  m, n, k_v));

  FFI_ASSIGN_OR_RETURN(auto workspace,
                       AllocateWorkspace<T>(scratch, lwork, "ormqr"));
  FFI_ASSIGN_OR_RETURN(auto info, AllocateWorkspace<int>(scratch, 1, "ormqr"));

  auto a_data = static_cast<T*>(a.untyped_data());
  auto tau_data = static_cast<T*>(tau.untyped_data());
  auto c_data = static_cast<T*>(c.untyped_data());
  auto out_data = static_cast<T*>(out->untyped_data());
  if (c_data != out_data) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        out_data, c_data, c.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  int64_t out_step = static_cast<int64_t>(m) * n;
  int64_t a_step = static_cast<int64_t>(a_rows) * a_cols;
  for (auto i = 0; i < batch; ++i) {
    FFI_RETURN_IF_ERROR_STATUS(solver::Ormqr<T>(
        handle.get(), side, trans, m, n, k_v, a_data, lda, tau_data,
        out_data, m, workspace, lwork, info));
    out_data += out_step;
    a_data += a_step;
    tau_data += k_v;
  }
  return ffi::Error::Success();
}

ffi::Error OrmqrDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                          ffi::AnyBuffer a, ffi::AnyBuffer tau,
                          ffi::AnyBuffer c, bool left, bool transpose,
                          ffi::Result<ffi::AnyBuffer> out) {
  auto dataType = a.element_type();
  if (dataType != tau.element_type() || dataType != c.element_type() ||
      dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "The inputs and outputs to ormqr must have the same element type");
  }
  FFI_ASSIGN_OR_RETURN((auto [batch, a_rows, a_cols]),
                       SplitBatch2D(a.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [tau_batch, k]),
                       SplitBatch1D(tau.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [c_batch, c_rows, c_cols]),
                       SplitBatch2D(c.dimensions()));
  if (tau_batch != batch || c_batch != batch) {
    return ffi::Error::InvalidArgument(
        "The batch dimensions of the inputs to ormqr must match");
  }
  FFI_RETURN_IF_ERROR(
      CheckShape(out->dimensions(), {batch, c_rows, c_cols}, "out", "ormqr"));
  SOLVER_DISPATCH_IMPL(OrmqrImpl, batch, c_rows, c_cols, k, a_rows, a_cols,
                       left, transpose, stream, scratch, a, tau, c, out);
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in ormqr", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OrmqrFfi, OrmqrDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // a
                                  .Arg<ffi::AnyBuffer>()  // tau
                                  .Arg<ffi::AnyBuffer>()  // c
                                  .Attr<bool>("left")
                                  .Attr<bool>("transpose")
                                  .Ret<ffi::AnyBuffer>()  // out
);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
