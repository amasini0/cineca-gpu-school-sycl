cmake_minimum_required(VERSION 3.20)
project(heat2D VERSION 0.0.1 LANGUAGES CXX)

# Check SYCL support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fsycl HAVE_SYCL)
if (NOT HAVE_SYCL)
    message(FATAL_ERROR "The selected CXX compiler does not support SYCL")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directory for binary
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Add subdirectory of source files
add_subdirectory(src)

