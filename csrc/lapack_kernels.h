// LAPACK ORMQR kernel declarations for jaxtra.
// Implements dormqr/sormqr/cunmqr/zunmqr via the XLA FFI typed API.
#pragma once

#include <complex>
#include <cstdint>

#include "xla/ffi/api/ffi.h"

namespace jaxtra {

namespace ffi = xla::ffi;

// Fortran LAPACK declarations (underscore suffix, LP64 ints).
extern "C" {
void sormqr_(char* side, char* trans, int* m, int* n, int* k, float* a,
             int* lda, float* tau, float* c, int* ldc, float* work,
             int* lwork, int* info);
void dormqr_(char* side, char* trans, int* m, int* n, int* k, double* a,
             int* lda, double* tau, double* c, int* ldc, double* work,
             int* lwork, int* info);
void cunmqr_(char* side, char* trans, int* m, int* n, int* k,
             std::complex<float>* a, int* lda, std::complex<float>* tau,
             std::complex<float>* c, int* ldc, std::complex<float>* work,
             int* lwork, int* info);
void zunmqr_(char* side, char* trans, int* m, int* n, int* k,
             std::complex<double>* a, int* lda, std::complex<double>* tau,
             std::complex<double>* c, int* ldc, std::complex<double>* work,
             int* lwork, int* info);
}  // extern "C"

// ---------------------------------------------------------------------------
// XLA FFI handlers (one per dtype).
// The kernel signature seen from JAX:
//   ormqr(a: T[..., m, k], tau: T[..., k], c: T[..., p, q], *,
//         left: bool, transpose: bool) -> T[..., p, q]
//
// • left=True  → apply Q on the left:  result = Q  @ c  (requires m == p)
//               apply Q.H on the left: result = Q.H @ c  when transpose=True
// • left=False → apply Q on the right: result = c  @ Q  (requires m == q)
//               apply Q.H on the right:result = c  @ Q.H when transpose=True
//
// The Householder vectors live in the lower-trapezoidal part of `a` (output
// of geqrf).  Taus are the k reflector scalars.
// ---------------------------------------------------------------------------

ffi::Error OrmqrF32(ffi::Buffer<ffi::DataType::F32> a,
                    ffi::Buffer<ffi::DataType::F32> tau,
                    ffi::Buffer<ffi::DataType::F32> c,
                    ffi::ResultBuffer<ffi::DataType::F32> c_out,
                    bool left, bool transpose);

ffi::Error OrmqrF64(ffi::Buffer<ffi::DataType::F64> a,
                    ffi::Buffer<ffi::DataType::F64> tau,
                    ffi::Buffer<ffi::DataType::F64> c,
                    ffi::ResultBuffer<ffi::DataType::F64> c_out,
                    bool left, bool transpose);

ffi::Error OrmqrC64(ffi::Buffer<ffi::DataType::C64> a,
                    ffi::Buffer<ffi::DataType::C64> tau,
                    ffi::Buffer<ffi::DataType::C64> c,
                    ffi::ResultBuffer<ffi::DataType::C64> c_out,
                    bool left, bool transpose);

ffi::Error OrmqrC128(ffi::Buffer<ffi::DataType::C128> a,
                     ffi::Buffer<ffi::DataType::C128> tau,
                     ffi::Buffer<ffi::DataType::C128> c,
                     ffi::ResultBuffer<ffi::DataType::C128> c_out,
                     bool left, bool transpose);

}  // namespace jaxtra
