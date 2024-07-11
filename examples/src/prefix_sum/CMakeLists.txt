# Add SYCL executable
add_executable(prefix_sum_sycl prefix_sum.cpp)
target_link_libraries(prefix_sum_sycl PRIVATE utils)
target_link_libraries(prefix_sum_sycl PRIVATE sycl)

# Add CUDA executable
add_executable(prefix_sum_cuda prefix_sum.cu)
target_link_libraries(prefix_sum_cuda PRIVATE utils)
