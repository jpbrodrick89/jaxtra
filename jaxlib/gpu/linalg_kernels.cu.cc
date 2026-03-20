/* Copyright 2025 The jaxtra Authors.

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

// LDL ipiv-to-permutation CUDA kernel.
// Converts a LAPACK Bunch-Kaufman pivot array (ipiv, 1-indexed) to a
// permutation array (0-indexed).  One CUDA thread per batch element runs the
// sequential swap loop entirely in-kernel — a single kernel launch regardless
// of n, avoiding O(n) kernel dispatches from host code.

#include <cstdint>
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

// This file is CUDA-only; use cudaStream_t and jax::cuda directly rather
// than pulling in vendor.h (which transitively includes cupti.h).

namespace jax {
namespace cuda {

namespace ffi = ::xla::ffi;

// One thread per batch element: blockIdx.x = batch index, threadIdx.x = 0.
__global__ void LdlIpivToPermutationKernel(
    const int32_t* ipiv, int32_t* perm, int64_t n, bool lower) {
  int b = blockIdx.x;
  const int32_t* piv = ipiv + b * n;
  int32_t* p = perm + b * n;
  for (int i = 0; i < (int)n; ++i) p[i] = i;
  if (lower) {
    int k = 0;
    while (k < (int)n) {
      int pk = piv[k];
      if (pk > 0) {
        int j = pk - 1;
        int t = p[k]; p[k] = p[j]; p[j] = t;
        k++;
      } else {
        int j = -pk - 1;
        int t = p[k + 1]; p[k + 1] = p[j]; p[j] = t;
        k += 2;
      }
    }
  } else {
    int k = (int)n - 1;
    while (k >= 0) {
      int pk = piv[k];
      if (pk > 0) {
        int j = pk - 1;
        int t = p[k]; p[k] = p[j]; p[j] = t;
        k--;
      } else {
        int j = -pk - 1;
        int t = p[k - 1]; p[k - 1] = p[j]; p[j] = t;
        k -= 2;
      }
    }
  }
}

ffi::Error LdlIpivToPermutationDispatch(
    cudaStream_t stream, ffi::AnyBuffer ipiv, bool lower,
    ffi::Result<ffi::AnyBuffer> perm_out) {
  auto dims = ipiv.dimensions();
  if (dims.size() < 1)
    return ffi::Error::InvalidArgument(
        "ldl_ipiv_to_permutation: ipiv must be >= 1D");
  int64_t n = dims.back();
  int64_t batch = 1;
  for (size_t i = 0; i + 1 < dims.size(); ++i) batch *= dims[i];
  auto* ipiv_ptr = static_cast<const int32_t*>(ipiv.untyped_data());
  auto* perm_ptr = static_cast<int32_t*>(perm_out->untyped_data());
  if (batch > 0 && n > 0)
    LdlIpivToPermutationKernel<<<(int)batch, 1, 0, stream>>>(
        ipiv_ptr, perm_ptr, n, lower);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    LdlIpivToPermutationFfi, LdlIpivToPermutationDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()   // ipiv (S32)
        .Attr<bool>("lower")
        .Ret<ffi::AnyBuffer>()   // perm (S32)
);

}  // namespace cuda
}  // namespace jax
