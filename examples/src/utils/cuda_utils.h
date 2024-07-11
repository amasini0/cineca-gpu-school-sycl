#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream>
#include <string_view>

// error handling function
static void HandleError(cudaError_t err, const char * file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

// error hadling macro
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


void print_kernel_info(std::string_view kernelName, const size_t size, const size_t device)
{   
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << std::endl;
    std::cout << "General info: " << std::endl;
    std::cout << "KERNEL: " << kernelName << std::endl;
    std::cout << "DEVICE: " << props.name << std::endl;
    std::cout << "DATA SIZE: " << size << std::endl;
    std::cout << std::endl;
}

void print_launch_info(const dim3& grid, const dim3& block, const size_t device)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    std::cout << "Launch parameters:" << std::endl;
    std::cout << "GRID  SIZE: (" << grid.x  << ", " << grid.y  << ")" << std::endl;
    std::cout << "BLOCK SIZE: (" << block.x << ", " << block.y << ")" << std::endl;
    std::cout << "WARP  SIZE: " << props.warpSize << std::endl;
    std::cout << std::endl;
}

#endif // CUDA_UTILS_H
