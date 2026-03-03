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

// XLA FFI handler for cuSPARSE gpsvInterleavedBatch (pentadiagonal solve).
//
// The Python lowering requests interleaved memory layouts via XLA FFI
// operand_layouts, so all buffers arrive in the interleaved-batch format
// expected by gpsvInterleavedBatch:
//   data[i * batchCount + j]   for element i of batch j
//
// batchCount is inferred from the diagonal shape via SplitBatch1D.

#include "jaxlib/gpu/sparse_kernels_ffi.h"

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/sparse_handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

// ---------------------------------------------------------------------------
// Type-erased dispatch helpers
// ---------------------------------------------------------------------------

// Buffer size query for gpsvInterleavedBatch.
template <typename T>
absl::StatusOr<size_t> GpsvBufferSize(gpusparseHandle_t handle, int m,
                                      int batchCount,
                                      const T* ds, const T* dl, const T* d,
                                      const T* du, const T* dw, const T* x);

template <>
absl::StatusOr<size_t> GpsvBufferSize<float>(
    gpusparseHandle_t handle, int m, int batchCount,
    const float* ds, const float* dl, const float* d,
    const float* du, const float* dw, const float* x) {
  size_t bufSize = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseSgpsvInterleavedBatch_bufferSizeExt(
          handle, /*algo=*/0, m,
          const_cast<float*>(ds), const_cast<float*>(dl),
          const_cast<float*>(d),  const_cast<float*>(du),
          const_cast<float*>(dw), const_cast<float*>(x),
          batchCount, &bufSize)));
  return bufSize;
}

template <>
absl::StatusOr<size_t> GpsvBufferSize<double>(
    gpusparseHandle_t handle, int m, int batchCount,
    const double* ds, const double* dl, const double* d,
    const double* du, const double* dw, const double* x) {
  size_t bufSize = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseDgpsvInterleavedBatch_bufferSizeExt(
          handle, /*algo=*/0, m,
          const_cast<double*>(ds), const_cast<double*>(dl),
          const_cast<double*>(d),  const_cast<double*>(du),
          const_cast<double*>(dw), const_cast<double*>(x),
          batchCount, &bufSize)));
  return bufSize;
}

template <>
absl::StatusOr<size_t> GpsvBufferSize<gpuComplex>(
    gpusparseHandle_t handle, int m, int batchCount,
    const gpuComplex* ds, const gpuComplex* dl, const gpuComplex* d,
    const gpuComplex* du, const gpuComplex* dw, const gpuComplex* x) {
  size_t bufSize = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseCgpsvInterleavedBatch_bufferSizeExt(
          handle, /*algo=*/0, m,
          const_cast<gpuComplex*>(ds), const_cast<gpuComplex*>(dl),
          const_cast<gpuComplex*>(d),  const_cast<gpuComplex*>(du),
          const_cast<gpuComplex*>(dw), const_cast<gpuComplex*>(x),
          batchCount, &bufSize)));
  return bufSize;
}

template <>
absl::StatusOr<size_t> GpsvBufferSize<gpuDoubleComplex>(
    gpusparseHandle_t handle, int m, int batchCount,
    const gpuDoubleComplex* ds, const gpuDoubleComplex* dl,
    const gpuDoubleComplex* d,  const gpuDoubleComplex* du,
    const gpuDoubleComplex* dw, const gpuDoubleComplex* x) {
  size_t bufSize = 0;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpusparseZgpsvInterleavedBatch_bufferSizeExt(
          handle, /*algo=*/0, m,
          const_cast<gpuDoubleComplex*>(ds), const_cast<gpuDoubleComplex*>(dl),
          const_cast<gpuDoubleComplex*>(d),  const_cast<gpuDoubleComplex*>(du),
          const_cast<gpuDoubleComplex*>(dw), const_cast<gpuDoubleComplex*>(x),
          batchCount, &bufSize)));
  return bufSize;
}

// Solve with gpsvInterleavedBatch.
template <typename T>
absl::Status GpsvSolve(gpusparseHandle_t handle, int m, int batchCount,
                       T* ds, T* dl, T* d, T* du, T* dw, T* x, void* buf);

template <>
absl::Status GpsvSolve<float>(gpusparseHandle_t handle, int m, int batchCount,
                               float* ds, float* dl, float* d, float* du,
                               float* dw, float* x, void* buf) {
  return JAX_AS_STATUS(gpusparseSgpsvInterleavedBatch(
      handle, /*algo=*/0, m, ds, dl, d, du, dw, x, batchCount, buf));
}

template <>
absl::Status GpsvSolve<double>(gpusparseHandle_t handle, int m, int batchCount,
                                double* ds, double* dl, double* d, double* du,
                                double* dw, double* x, void* buf) {
  return JAX_AS_STATUS(gpusparseDgpsvInterleavedBatch(
      handle, /*algo=*/0, m, ds, dl, d, du, dw, x, batchCount, buf));
}

template <>
absl::Status GpsvSolve<gpuComplex>(gpusparseHandle_t handle, int m,
                                    int batchCount,
                                    gpuComplex* ds, gpuComplex* dl,
                                    gpuComplex* d,  gpuComplex* du,
                                    gpuComplex* dw, gpuComplex* x, void* buf) {
  return JAX_AS_STATUS(gpusparseCgpsvInterleavedBatch(
      handle, /*algo=*/0, m, ds, dl, d, du, dw, x, batchCount, buf));
}

template <>
absl::Status GpsvSolve<gpuDoubleComplex>(
    gpusparseHandle_t handle, int m, int batchCount,
    gpuDoubleComplex* ds, gpuDoubleComplex* dl,
    gpuDoubleComplex* d,  gpuDoubleComplex* du,
    gpuDoubleComplex* dw, gpuDoubleComplex* x, void* buf) {
  return JAX_AS_STATUS(gpusparseZgpsvInterleavedBatch(
      handle, /*algo=*/0, m, ds, dl, d, du, dw, x, batchCount, buf));
}

// ---------------------------------------------------------------------------
// GpsvImpl — typed FFI implementation
// ---------------------------------------------------------------------------

template <typename T>
ffi::Error GpsvImpl(int64_t batch, int64_t m, int64_t nrhs,
                    gpuStream_t stream,
                    ffi::ScratchAllocator& scratch,
                    ffi::AnyBuffer ds, ffi::AnyBuffer dl, ffi::AnyBuffer d,
                    ffi::AnyBuffer du, ffi::AnyBuffer dw, ffi::AnyBuffer b,
                    ffi::Result<ffi::AnyBuffer> out) {
  FFI_ASSIGN_OR_RETURN(auto m_v, MaybeCastNoOverflow<int>(m));
  FFI_ASSIGN_OR_RETURN(auto batch_v, MaybeCastNoOverflow<int>(batch));

  FFI_ASSIGN_OR_RETURN(auto handle, SparseHandlePool::Borrow(stream));

  auto x_ptr  = static_cast<T*>(out->untyped_data());

  // Copy b -> out if they differ.
  auto b_ptr  = static_cast<const T*>(b.untyped_data());
  if (b_ptr != x_ptr) {
    JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpyAsync(
        x_ptr, b_ptr, b.size_bytes(), gpuMemcpyDeviceToDevice, stream));
  }

  FFI_ASSIGN_OR_RETURN(size_t buf_bytes,
                       GpsvBufferSize<T>(handle.get(), m_v, batch_v,
                                         static_cast<const T*>(ds.untyped_data()),
                                         static_cast<const T*>(dl.untyped_data()),
                                         static_cast<const T*>(d.untyped_data()),
                                         static_cast<const T*>(du.untyped_data()),
                                         static_cast<const T*>(dw.untyped_data()),
                                         x_ptr));

  // Allocate workspace from XLA scratch allocator.
  FFI_ASSIGN_OR_RETURN(auto buf,
                       AllocateWorkspace<char>(scratch,
                                               (buf_bytes + sizeof(char) - 1) / sizeof(char),
                                               "gpsv"));

  // Buffers arrive in interleaved-batch layout (dim 0 = batch, stride 1),
  // so cuSPARSE handles all batch elements in a single call.
  // gpsvInterleavedBatch solves one RHS per call, so we loop over nrhs.
  // algo=0 (QR) does not modify diagonal inputs, so reuse across nrhs is safe.
  auto ds_data = static_cast<T*>(ds.untyped_data());
  auto dl_data = static_cast<T*>(dl.untyped_data());
  auto d_data  = static_cast<T*>(d.untyped_data());
  auto du_data = static_cast<T*>(du.untyped_data());
  auto dw_data = static_cast<T*>(dw.untyped_data());

  for (int64_t j = 0; j < nrhs; ++j) {
    FFI_RETURN_IF_ERROR_STATUS(GpsvSolve<T>(
        handle.get(), m_v, batch_v,
        ds_data, dl_data, d_data, du_data, dw_data,
        x_ptr + j * batch * m,
        buf));
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// GpsvDispatch — type dispatch
// ---------------------------------------------------------------------------

ffi::Error GpsvDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                         ffi::AnyBuffer ds, ffi::AnyBuffer dl, ffi::AnyBuffer d,
                         ffi::AnyBuffer du, ffi::AnyBuffer dw, ffi::AnyBuffer b,
                         ffi::Result<ffi::AnyBuffer> out) {
  auto dataType = d.element_type();
  if (dataType != ds.element_type() || dataType != dl.element_type() ||
      dataType != du.element_type() || dataType != dw.element_type() ||
      dataType != b.element_type() || dataType != out->element_type()) {
    return ffi::Error::InvalidArgument(
        "All inputs and output of pentadiagonal_solve must have the same element type");
  }

  FFI_ASSIGN_OR_RETURN((auto [batch, m]), SplitBatch1D(d.dimensions()));
  FFI_RETURN_IF_ERROR(CheckShape(ds.dimensions(), {batch, m}, "ds", "gpsv"));
  FFI_RETURN_IF_ERROR(CheckShape(dl.dimensions(), {batch, m}, "dl", "gpsv"));
  FFI_RETURN_IF_ERROR(CheckShape(du.dimensions(), {batch, m}, "du", "gpsv"));
  FFI_RETURN_IF_ERROR(CheckShape(dw.dimensions(), {batch, m}, "dw", "gpsv"));
  // b has shape (..., m, nrhs) — use SplitBatch2D to extract nrhs.
  FFI_ASSIGN_OR_RETURN((auto [b_batch, b_rows, nrhs]), SplitBatch2D(b.dimensions()));
  FFI_ASSIGN_OR_RETURN((auto [out_batch, out_rows, out_nrhs]), SplitBatch2D(out->dimensions()));

  switch (dataType) {
    case ffi::F32:
      return GpsvImpl<float>(batch, m, nrhs, stream, scratch,
                             ds, dl, d, du, dw, b, out);
    case ffi::F64:
      return GpsvImpl<double>(batch, m, nrhs, stream, scratch,
                              ds, dl, d, du, dw, b, out);
    case ffi::C64:
      return GpsvImpl<gpuComplex>(batch, m, nrhs, stream, scratch,
                                   ds, dl, d, du, dw, b, out);
    case ffi::C128:
      return GpsvImpl<gpuDoubleComplex>(batch, m, nrhs, stream, scratch,
                                        ds, dl, d, du, dw, b, out);
    default:
      break;
  }
  return ffi::Error::InvalidArgument(absl::StrFormat(
      "Unsupported dtype %s in gpsv", absl::FormatStreamed(dataType)));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GpsvFfi, GpsvDispatch,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // ds
                                  .Arg<ffi::AnyBuffer>()  // dl
                                  .Arg<ffi::AnyBuffer>()  // d
                                  .Arg<ffi::AnyBuffer>()  // du
                                  .Arg<ffi::AnyBuffer>()  // dw
                                  .Arg<ffi::AnyBuffer>()  // b
                                  .Ret<ffi::AnyBuffer>()  // out (x)
);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
