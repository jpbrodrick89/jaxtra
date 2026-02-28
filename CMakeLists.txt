cmake_minimum_required(VERSION 3.20)
project(jaxtra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Python + pybind11
# ---------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

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
# by jaxtra._core using ctypes on scipy.linalg.cython_lapack.__pyx_capi__.
# This mirrors jaxlib's initialize() mechanism and keeps the wheel tiny.
pybind11_add_module(_jaxtra MODULE
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
