#ifndef SYCL_UTILS_H
#define SYCL_UTILS_H

#include <sycl/sycl.hpp>
#include <iostream>
#include <string_view>

void print_kernel_info(std::string_view kernelName, const size_t size, const sycl::queue& Q)
{ 
    std::cout << std::endl;
    std::cout << "General info: " << std::endl;
    std::cout << "KERNEL: " << kernelName << std::endl;
    std::cout << "DEVICE: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "DATA SIZE: " << size << std::endl;
    std::cout << std::endl;
}

template <int D1, int D2> 
void print_launch_info(const sycl::range<D1>& grid, const sycl::range<D2>& block, const sycl::queue& Q)
{  
    std::cout << "Launch parameters:" << std::endl;
    std::cout << "GRID SIZE: ( ";
    for (unsigned int i{}; i < D1; ++i) {
        std::cout << grid[i] << " ";
    }
    std::cout << ")" << std::endl;
    std::cout << "WORKGROUP SIZE: ( "; 
    for (unsigned int i{}; i < D2; ++i) {
        std::cout << block[i] << " ";
    }
    std::cout << ")" << std::endl;
    std::cout << "SUBGROUP SIZES: [ ";
    auto sg_sizes{ Q.get_device().get_info<sycl::info::device::sub_group_sizes>() };
    for (auto size: sg_sizes) std::cout << size << " ";
    std::cout << "]" << std::endl;
    std::cout << std::endl;
}

#endif // SYCL_UTILS_H
