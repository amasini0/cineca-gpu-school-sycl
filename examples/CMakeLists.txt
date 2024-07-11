cmake_minimum_required(VERSION 3.20)
project(gpu_benchmarks LANGUAGES CXX CUDA)

# Chack sycl support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fsycl HAVE_SYCL)
if(NOT HAVE_SYCL)
    message(FATAL_ERROR "The selected CXX compiler does not support SYCL")
endif()

# Configure CXX options
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configure CUDA options
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set output directory for all compiled binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Create stub target to port compile/link options
add_library(sycl INTERFACE)
set(SYCL_TARGETS "spir64_x86_64-unknown-unkwnown,nvptx64-nvidia-cuda")
target_compile_options(sycl INTERFACE
    -fsycl -fsycl-targets=${SYCL_TARGETS}
    -Xclang -opaque-pointers)
target_link_options(sycl INTERFACE
    -fsycl -fsycl-targets=${SYCL_TARGETS})

# Add subdirectory with source files
add_subdirectory(src)
