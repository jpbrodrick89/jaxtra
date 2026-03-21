cmake_minimum_required(VERSION 3.20)

# JAXTRA_CUDA: ON = require CUDA, OFF = skip GPU extension, auto = detect (default).
# Set via cmake define (-DJAXTRA_CUDA=ON/OFF) or the JAXTRA_CUDA env var (via pyproject.toml).
# GPU extension uses one CUDA kernel file (jaxlib/gpu/linalg_kernels.cu.cc); CUDA language
# is enabled inside the if(_jaxtra_cuda_build) block when building that extension.
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
  jaxlib/cpu/jaxtra_module.cc
  jaxlib/cpu/lapack_kernels.cc
)

target_include_directories(_jaxtra PRIVATE
  ${XLA_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/jaxlib/cpu
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
#   jaxlib/gpu/solver_kernels_ffi.cc  → jaxlib/gpu/solver_kernels_ffi.cc
#   jaxlib/gpu/solver_interface.cc    → jaxlib/gpu/solver_interface.cc
#   jaxlib/gpu/gpu_kernel_helpers.cc  → jaxlib/gpu/gpu_kernel_helpers.cc
#   jaxlib/gpu/solver_handle_pool.cc  → jaxlib/gpu/solver_handle_pool.cc
#   jaxlib/gpu/solver.cc + gpu_kernels.cc → jaxlib/cuda/jaxtra_cuda_module.cc
#
# Bundled jaxlib headers (jaxlib/gpu/, jaxlib/ffi_helpers.h) replace Bazel's @jax//jaxlib deps.
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

  # Enable CUDA language for compiling __global__ kernels (linalg_kernels.cu.cc).
  # Must come before any target that uses CUDA sources.
  set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Abseil 20260107.1 requires C++20; bump the standard here so CUDA toolkit
  # detection (above) ran with C++17, which FindCUDAToolkit/FindThreads needs.
  # Also set CMAKE_CXX_FLAGS directly so abseil's check_cxx_source_compiles
  # (which runs in a subdirectory scope) reliably sees the C++20 standard.
  set(CMAKE_CXX_STANDARD 20)
  string(APPEND CMAKE_CXX_FLAGS " -std=gnu++20")
  # Pre-seed abseil's C++ standard check results so AbseilDll.cmake does not
  # re-run check_cxx_source_compiles (which fails to see C++20 flags in the
  # FetchContent subdirectory scope on this system despite GCC 14 supporting it).
  set(ABSL_INTERNAL_AT_LEAST_CXX17 1 CACHE INTERNAL "" FORCE)
  set(ABSL_INTERNAL_AT_LEAST_CXX20 1 CACHE INTERNAL "" FORCE)

  # --- Abseil (Bazel uses @com_google_absl; here we use FetchContent) ---
  find_package(absl QUIET)
  if(NOT absl_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      abseil-cpp
      GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
      GIT_TAG        20260107.1
      GIT_SHALLOW    TRUE
    )
    set(ABSL_PROPAGATE_CXX_STD ON)
    set(BUILD_TESTING OFF)
    FetchContent_MakeAvailable(abseil-cpp)
  endif()

  # --- LDL kernel OBJECT library (compiled by nvcc via LANGUAGE CUDA) ---
  # .cu.cc extension is not auto-detected by CMake; set_source_files_properties
  # tells CMake to compile this file with the CUDA compiler (nvcc).
  add_library(jaxtra_linalg_kernels OBJECT
    jaxlib/gpu/linalg_kernels.cu.cc)
  set_source_files_properties(jaxlib/gpu/linalg_kernels.cu.cc
    PROPERTIES LANGUAGE CUDA)
  set_target_properties(jaxtra_linalg_kernels PROPERTIES
    CUDA_STANDARD 20
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(jaxtra_linalg_kernels PRIVATE JAX_GPU_CUDA)
  target_include_directories(jaxtra_linalg_kernels PRIVATE
    ${XLA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}      # resolves jaxlib/gpu/vendor.h
    ${CUDAToolkit_INCLUDE_DIRS})

  # --- _jaxtra_cuda module ---
  nanobind_add_module(_jaxtra_cuda MODULE
    jaxlib/cuda/jaxtra_cuda_module.cc
    jaxlib/gpu/solver_kernels_ffi.cc
    jaxlib/gpu/solver_interface.cc
    jaxlib/gpu/gpu_kernel_helpers.cc
    jaxlib/gpu/solver_handle_pool.cc
    jaxlib/gpu/sparse_kernels_ffi.cc
    jaxlib/gpu/sparse_handle_pool.cc
    $<TARGET_OBJECTS:jaxtra_linalg_kernels>
  )

  target_compile_definitions(_jaxtra_cuda PRIVATE JAX_GPU_CUDA)

  # Find cuDNN and CUPTI headers via nvidia-* packages bundled with jax[cuda13].
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
      "import nvidia.cudnn, os; print(os.path.join(nvidia.cudnn.__path__[0], 'include'))"
    OUTPUT_VARIABLE CUDNN_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cudnn_result
  )
  if(NOT _cudnn_result EQUAL 0 OR NOT EXISTS "${CUDNN_INCLUDE_DIR}/cudnn.h")
    message(FATAL_ERROR
      "cuDNN header not found, make sure you have cuda-compatible version of jax installed (e.g. jax[cuda13])")
  endif()
  message(STATUS "cuDNN include dir: ${CUDNN_INCLUDE_DIR}")

  # CUDAToolkit_CUPTI_INCLUDE_DIR may contain relative path components; resolve it.
  get_filename_component(CUPTI_INCLUDE_DIR "${CUDAToolkit_CUPTI_INCLUDE_DIR}" ABSOLUTE)
  message(STATUS "CUPTI include dir: ${CUPTI_INCLUDE_DIR}")

  target_include_directories(_jaxtra_cuda PRIVATE
    ${XLA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}            # resolves "jaxlib/gpu/..." bundled headers
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CUPTI_INCLUDE_DIR}
  )

  target_compile_options(_jaxtra_cuda PRIVATE
    -Wno-sign-conversion
    -Wno-unused-parameter
  )

  target_link_libraries(_jaxtra_cuda PRIVATE
    CUDA::cudart
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

  # --- _jaxtra_hybrid module ---
  # Hybrid kernels: LAPACK hetrf/hetrs on CPU, dispatched via CUDA FFI.
  # LAPACK pointers loaded from scipy at import time (no LAPACK link needed).
  nanobind_add_module(_jaxtra_hybrid MODULE
    jaxlib/cuda/jaxtra_hybrid_module.cc
    jaxlib/gpu/hybrid_kernels_ffi.cc
    jaxlib/gpu/gpu_kernel_helpers.cc
  )

  target_compile_definitions(_jaxtra_hybrid PRIVATE JAX_GPU_CUDA)

  target_include_directories(_jaxtra_hybrid PRIVATE
    ${XLA_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${CUPTI_INCLUDE_DIR}
  )

  target_compile_options(_jaxtra_hybrid PRIVATE
    -Wno-sign-conversion
    -Wno-unused-parameter
  )

  target_link_libraries(_jaxtra_hybrid PRIVATE
    CUDA::cudart
    absl::status
    absl::statusor
    absl::str_format
    absl::strings
    absl::log
  )

  install(TARGETS _jaxtra_hybrid LIBRARY DESTINATION jaxtra)
  message(STATUS "Building _jaxtra_hybrid (hybrid CPU-LAPACK / GPU-dispatch extension)")
endif()
