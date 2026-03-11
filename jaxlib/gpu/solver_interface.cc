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

// Verbatim from JAX PR #35104: jaxlib/gpu/solver_interface.cc (ormqr section).
// Only the ormqr/unmqr specialisations are included; all other routines are
// provided by the installed jaxlib GPU shared library.
//
// Non-Bazel difference: "third_party/gpus/cuda/include/cusolverSp.h" is
// replaced by the standard CMake CUDA toolkit path <cusolverSp.h>.

#include "jaxlib/gpu/solver_interface.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"

#ifdef JAX_GPU_CUDA
#include <cusolverSp.h>  // non-Bazel: was "third_party/gpus/cuda/include/cusolverSp.h"
#endif

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace solver {

// Householder multiply: ormqr/unmqr

#define JAX_GPU_DEFINE_ORMQR(Type, Name)                                       \
  template <>                                                                  \
  absl::StatusOr<int> OrmqrBufferSize<Type>(                                   \
      gpusolverDnHandle_t handle, gpublasSideMode_t side,                      \
      gpublasOperation_t trans, int m, int n, int k) {                         \
    int lwork;                                                                 \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(Name##_bufferSize(                       \
        handle, side, trans, m, n, k, /*A=*/nullptr, /*lda=*/m,                \
        /*tau=*/nullptr, /*C=*/nullptr, /*ldc=*/m, &lwork)));                 \
    return lwork;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  absl::Status Ormqr<Type>(                                                    \
      gpusolverDnHandle_t handle, gpublasSideMode_t side,                      \
      gpublasOperation_t trans, int m, int n, int k, Type *a, int lda,         \
      Type *tau, Type *c, int ldc, Type *workspace, int lwork, int *info) {    \
    return JAX_AS_STATUS(                                                      \
        Name(handle, side, trans, m, n, k, a, lda, tau, c, ldc,                \
             workspace, lwork, info));                                          \
  }

JAX_GPU_DEFINE_ORMQR(float, gpusolverDnSormqr);
JAX_GPU_DEFINE_ORMQR(double, gpusolverDnDormqr);
JAX_GPU_DEFINE_ORMQR(gpuComplex, gpusolverDnCunmqr);
JAX_GPU_DEFINE_ORMQR(gpuDoubleComplex, gpusolverDnZunmqr);
#undef JAX_GPU_DEFINE_ORMQR

// Symmetric/complex-symmetric indefinite factorization: sytrf
// cuSolver provides all four dtype variants (S/D/C/Z).

#ifdef JAX_GPU_CUDA
#define JAX_GPU_DEFINE_SYTRF(Type, Name)                                      \
  template <>                                                                 \
  absl::StatusOr<int> SytrfBufferSize<Type>(gpusolverDnHandle_t handle,      \
                                            int n, Type *a, int lda) {       \
    int lwork;                                                                \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(Name##_bufferSize(handle, n, a, lda,   \
                                                         &lwork)));           \
    return lwork;                                                             \
  }                                                                           \
  template <>                                                                 \
  absl::Status Sytrf<Type>(gpusolverDnHandle_t handle,                       \
                            gpusolverFillMode_t uplo, int n, Type *a,        \
                            int lda, int *ipiv, Type *work, int lwork,       \
                            int *info) {                                      \
    return JAX_AS_STATUS(                                                     \
        Name(handle, uplo, n, a, lda, ipiv, work, lwork, info));             \
  }

JAX_GPU_DEFINE_SYTRF(float,           gpusolverDnSsytrf);
JAX_GPU_DEFINE_SYTRF(double,          gpusolverDnDsytrf);
JAX_GPU_DEFINE_SYTRF(gpuComplex,      gpusolverDnCsytrf);
JAX_GPU_DEFINE_SYTRF(gpuDoubleComplex, gpusolverDnZsytrf);
#undef JAX_GPU_DEFINE_SYTRF
#endif  // JAX_GPU_CUDA

}  // namespace solver
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
