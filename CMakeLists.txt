cmake_minimum_required(VERSION 3.20)

# JAXTRA_CUDA: ON = require CUDA, OFF = skip GPU extension, auto = detect (default).
# Set via cmake define (-DJAXTRA_CUDA=ON/OFF) or the JAXTRA_CUDA env var (via pyproject.toml).
# The GPU source files are host code only (no __global__ kernels), so LANGUAGES CUDA is not needed.
set(JAXTRA_CUDA "auto" CACHE STRING "Build CUDA GPU extension (_jaxtra_cuda): ON | OFF | auto")

project(jaxtra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# ---------------------------------------------------------------------------
# nanobind (same binding library as jaxlib; cmake dir via Python)
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _nb_result
)
if(NOT _nb_result EQUAL 0)
  message(FATAL_ERROR
    "Could not locate nanobind cmake dir. "
    "Install with: pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# XLA FFI headers (shipped with jaxlib)
# ---------------------------------------------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _jax_ffi_result
)
if(NOT _jax_ffi_result EQUAL 0)
  message(FATAL_ERROR
    "Could not locate XLA FFI headers via jax.ffi.include_dir(). "
    "Make sure jax is installed in the build environment: pip install jax")
endif()
message(STATUS "XLA FFI include dir: ${XLA_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# Extension module: _jaxtra
# ---------------------------------------------------------------------------
# No LAPACK link: LAPACK function pointers are resolved at Python import time
# by initialize() using nb::capsule::data() on scipy.linalg.cython_lapack,
# mirroring jaxlib's GetLapackKernelsFromScipy() exactly.
nanobind_add_module(_jaxtra MODULE
  csrc/jaxtra_module.cc
  csrc/lapack_kernels.cc
)

target_include_directories(_jaxtra PRIVATE
  ${XLA_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

# Suppress sign-conversion warnings from the XLA FFI headers.
target_compile_options(_jaxtra PRIVATE
  -Wno-sign-conversion
  -Wno-unused-parameter
)

# Install next to the jaxtra Python package so it can be imported.
install(TARGETS _jaxtra LIBRARY DESTINATION jaxtra)

# ---------------------------------------------------------------------------
# GPU extension: _jaxtra_cuda  (auto-built when CUDA is detected)
# ---------------------------------------------------------------------------
# Non-Bazel equivalents of the jaxlib GPU build rules from PR #35104:
#   jaxlib/gpu/solver_kernels_ffi.cc  → csrc/gpu/jaxlib/gpu/solver_kernels_ffi.cc
#   jaxlib/gpu/solver_interface.cc    → csrc/gpu/jaxlib/gpu/solver_interface.cc
#   jaxlib/gpu/gpu_kernel_helpers.cc  → csrc/gpu/jaxlib/gpu/gpu_kernel_helpers.cc
#   jaxlib/gpu/solver_handle_pool.cc  → csrc/gpu/jaxlib/gpu/solver_handle_pool.cc
#   jaxlib/gpu/solver.cc + gpu_kernels.cc → csrc/jaxtra_cuda_module.cc
#
# Bundled jaxlib headers (csrc/gpu/) replace Bazel's @jax//jaxlib deps.
# Abseil is fetched via FetchContent (Bazel uses @com_google_absl).
# CUDA headers are found via CMake CUDAToolkit (Bazel uses third_party/gpus).
if(JAXTRA_CUDA STREQUAL "auto")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(_jaxtra_cuda_build ON)
    message(STATUS "CUDA ${CUDAToolkit_VERSION} auto-detected: building _jaxtra_cuda")
  else()
    set(_jaxtra_cuda_build OFF)
    message(STATUS "CUDA not found: skipping _jaxtra_cuda (install jaxtra[gpu] with CUDA toolkit to enable)")
  endif()
elseif(JAXTRA_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(_jaxtra_cuda_build ON)
else()
  set(_jaxtra_cuda_build OFF)
  message(STATUS "CUDA GPU extension disabled (JAXTRA_CUDA=OFF)")
endif()

if(_jaxtra_cuda_build)
  # CUDAToolkit already found in the detection block above.

  # --- Abseil (Bazel uses @com_google_absl; here we use FetchContent) ---
  find_package(absl QUIET)
  if(NOT absl_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      abseil-cpp
      GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
      GIT_TAG        20240722.0
      GIT_SHALLOW    TRUE
    )
    set(ABSL_PROPAGATE_CXX_STD ON)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(abseil-cpp)
  endif()

  # --- _jaxtra_cuda module ---
  nanobind_add_module(_jaxtra_cuda MODULE
    csrc/jaxtra_cuda_module.cc
    csrc/gpu/jaxlib/gpu/solver_kernels_ffi.cc
    csrc/gpu/jaxlib/gpu/solver_interface.cc
    csrc/gpu/jaxlib/gpu/gpu_kernel_helpers.cc
    csrc/gpu/jaxlib/gpu/solver_handle_pool.cc
    csrc/gpu/jaxlib/gpu/sparse_kernels_ffi.cc
    csrc/gpu/jaxlib/gpu/sparse_handle_pool.cc
  )

  target_compile_definitions(_jaxtra_cuda PRIVATE JAX_GPU_CUDA)

  target_include_directories(_jaxtra_cuda PRIVATE
    ${XLA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/gpu   # resolves "jaxlib/gpu/..." bundled headers
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_compile_options(_jaxtra_cuda PRIVATE
    -Wno-sign-conversion
    -Wno-unused-parameter
  )

  target_link_libraries(_jaxtra_cuda PRIVATE
    CUDA::cusolver
    CUDA::cusparse
    absl::status
    absl::statusor
    absl::str_format
    absl::strings
    absl::log
    absl::synchronization
  )

  install(TARGETS _jaxtra_cuda LIBRARY DESTINATION jaxtra)
  message(STATUS "Building _jaxtra_cuda (CUDA GPU extension)")
endif()
