# Add SYCL executable
add_executable(matmul_global_sycl matmul_global.cpp)
target_link_libraries(matmul_global_sycl PRIVATE utils)
target_link_libraries(matmul_global_sycl PRIVATE sycl)

# Add CUDA executable
add_executable(matmul_global_cuda matmul_global.cu)
target_link_libraries(matmul_global_cuda PRIVATE utils)

