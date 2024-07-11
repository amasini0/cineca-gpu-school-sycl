# Create shared utils library
add_library(utils INTERFACE)
target_include_directories(utils INTERFACE utils)

# Add subdirs for benchmark binaries
add_subdirectory(matmul_global)
add_subdirectory(matmul_tiled)
add_subdirectory(prefix_sum)
add_subdirectory(stencil)
