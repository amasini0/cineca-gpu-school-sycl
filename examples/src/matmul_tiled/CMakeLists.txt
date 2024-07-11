# Add SYCL executable
add_executable(matmul_tiled_sycl matmul_tiled.cpp)
target_link_libraries(matmul_tiled_sycl PRIVATE utils)
target_link_libraries(matmul_tiled_sycl PRIVATE sycl)

# Add CUDA executable
add_executable(matmul_tiled_cuda matmul_tiled.cu)
target_link_libraries(matmul_tiled_cuda PRIVATE utils)
