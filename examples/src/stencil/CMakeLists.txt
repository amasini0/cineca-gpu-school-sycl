# Add SYCL executable
add_executable(stencil_sycl stencil.cpp)
target_link_libraries(stencil_sycl PRIVATE utils)
target_link_libraries(stencil_sycl PRIVATE sycl)

# Add CUDA executable
add_executable(stencil_cuda stencil.cu)
target_link_libraries(stencil_cuda PRIVATE utils)
