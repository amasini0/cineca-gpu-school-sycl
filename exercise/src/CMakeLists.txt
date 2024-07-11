# Serial CPU code
add_executable(heat2d_serial heat2d_serial.cpp)

# SYCL GPU code
add_executable(heat2d_sycl heat2d_sycl.cpp)
target_compile_options(heat2d_sycl PRIVATE -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -Xclang -opaque-pointers)
target_link_options(heat2d_sycl PRIVATE -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda -Xclang -opaque-pointers)

# Validation
add_executable(validate validate.cpp)
