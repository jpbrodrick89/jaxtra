// jaxlib/gpu/hybrid_kernels_ffi.h — LAPACK hetrf on CPU, GPU-dispatched.
//
// These kernels run on the host CPU (calling LAPACK) but are registered as
// CUDA FFI targets.  They synchronise the GPU stream, copy data device→host,
// call LAPACK, then copy host→device.  This mirrors jaxlib's cuhybrid_geqp3
// pattern and is used as a CPU fallback when cuSOLVER has no hetrf.
//
// Only complex types are provided (shetrf/dhetrf do not exist in LAPACK; real
// symmetric matrices use the sytrf path which cuSOLVER does support).
#pragma once

#include <complex>
#include <cstdint>

#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = ::xla::ffi;

// ---------------------------------------------------------------------------
// HybridHetDecomp<dtype> — LAPACK ?hetrf on CPU
// ---------------------------------------------------------------------------
template <ffi::DataType dtype>
struct HybridHetDecomp {
  using ValueType = ffi::NativeType<dtype>;

  // Fortran LAPACK calling convention for chetrf / zhetrf.
  using FnType = void(char* /*uplo*/, int* /*n*/, ValueType* /*a*/,
                      int* /*lda*/, int* /*ipiv*/, ValueType* /*work*/,
                      int* /*lwork*/, int* /*info*/);

  // Set by initialize() in the nanobind module at import time.
  inline static FnType* fn = nullptr;

  static ffi::Error Kernel(gpuStream_t stream,
                            ffi::Buffer<dtype> a, bool lower,
                            ffi::ResultBuffer<dtype> a_out,
                            ffi::ResultBuffer<ffi::DataType::S32> ipiv_out);
};

extern template struct HybridHetDecomp<ffi::DataType::C64>;
extern template struct HybridHetDecomp<ffi::DataType::C128>;

// ---------------------------------------------------------------------------
// Handler declarations
// ---------------------------------------------------------------------------
XLA_FFI_DECLARE_HANDLER_SYMBOL(cuhybrid_chetrf_ffi);
XLA_FFI_DECLARE_HANDLER_SYMBOL(cuhybrid_zhetrf_ffi);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
