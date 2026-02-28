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
# LAPACK / BLAS
# ---------------------------------------------------------------------------
# Strategy:
#  1. Try the standard CMake finder.
#  2. Fall back to finding OpenBLAS directly.
#  3. Add libgfortran RPATH from Python's scipy/numpy bundles if present,
#     so the extension is self-contained without a system gfortran install.
find_package(LAPACK QUIET)
if(NOT LAPACK_FOUND)
  message(STATUS "CMake FindLAPACK failed — searching for OpenBLAS directly")
  find_library(OPENBLAS_LIB
    NAMES openblas openblasp openblaso
    HINTS
      /usr/lib/x86_64-linux-gnu/openblas-pthread
      /usr/lib/x86_64-linux-gnu/openblas
      /usr/local/lib
      /opt/homebrew/opt/openblas/lib
      /usr/lib
    DOC "OpenBLAS library (includes LAPACK)"
  )
  if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    set(LAPACK_LIBRARIES "${OPENBLAS_LIB}")
  else()
    set(LAPACK_LIBRARIES "-lopenblas")
    message(WARNING "OpenBLAS not found; linking with -lopenblas")
  endif()
endif()
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")

# Find libgfortran shipped with scipy/numpy so we don't need system gfortran.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import os, glob, scipy; \
     base=os.path.dirname(os.path.dirname(scipy.__file__)); \
     hits=glob.glob(os.path.join(base,'scipy.libs','libgfortran*.so*'))+\
          glob.glob(os.path.join(base,'numpy.libs','libgfortran*.so*')); \
     print(os.path.dirname(hits[0]) if hits else '')"
  OUTPUT_VARIABLE GFORTRAN_RPATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(GFORTRAN_RPATH)
  message(STATUS "libgfortran RPATH from Python bundle: ${GFORTRAN_RPATH}")
  list(APPEND CMAKE_INSTALL_RPATH "${GFORTRAN_RPATH}")
  set(CMAKE_BUILD_RPATH "${GFORTRAN_RPATH}")
  # Also find the actual libgfortran .so for explicit linking to resolve
  # missing symbol errors on systems where system gfortran is absent.
  file(GLOB _GFORTRAN_LIBS "${GFORTRAN_RPATH}/libgfortran*.so*")
  foreach(_f ${_GFORTRAN_LIBS})
    if(NOT _f MATCHES "\\.so\\.[0-9]+\\.[0-9]+$")
      list(APPEND LAPACK_LIBRARIES "${_f}")
      break()
    endif()
  endforeach()
endif()

# Keep the RPATH in installed binaries (important for wheel repacking).
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---------------------------------------------------------------------------
# Extension module: _jaxtra
# ---------------------------------------------------------------------------
pybind11_add_module(_jaxtra MODULE
  csrc/jaxtra_module.cc
  csrc/lapack_kernels.cc
)

target_include_directories(_jaxtra PRIVATE
  ${XLA_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_link_libraries(_jaxtra PRIVATE ${LAPACK_LIBRARIES})

# Suppress sign-conversion warnings from the XLA FFI headers.
target_compile_options(_jaxtra PRIVATE
  -Wno-sign-conversion
  -Wno-unused-parameter
)

# Install next to the jaxtra Python package so it can be imported.
install(TARGETS _jaxtra LIBRARY DESTINATION jaxtra)
