#include "common_utils.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 };
constexpr size_t BLOCK{  32 };

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

__global__ void globalMatMul(const float* A, const float* B, float* C)
{
    const size_t row{ blockIdx.y * blockDim.y + threadIdx.y };
    const size_t col{ blockIdx.x * blockDim.x + threadIdx.x };

    if ((row >= SIZE) || (col >= SIZE)) return;

    float c{};
    for (size_t k{}; k < SIZE; ++k) {
        c += A[row * SIZE + k] * B[k * SIZE + col];
    }
    C[row * SIZE + col] = c;
}

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };

    // define thread block size and grid
    const auto nblocks{ static_cast<unsigned>(std::ceil(static_cast<float>(SIZE)/BLOCK)) };
    const dim3 grid{ nblocks, nblocks };
    const dim3 block{ BLOCK, BLOCK };

    // print kernel and launch info
    print_kernel_info("Global Matrix Multiplication CUDA", SIZE, 0);
    print_launch_info(grid, block, 0);

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    cudaEvent_t kernelEnd;
    cudaEventCreate(&kernelEnd);
    std::vector<float> timings{};
    timings.reserve(RUNS);

    // allocate host matrices
    float* hstA{ new float[nelems] };
    float* hstB{ new float[nelems] };
    float* hstC{ new float[nelems] };

    // initialize host matrices
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            hstA[i * SIZE + j] =  10.0 * i + j;
            hstB[i * SIZE + j] = (i == j) ? 1.0 : 0.0;
            hstC[i * SIZE + j] = -42.0;
        }
    }

    // allocate device matrices
    void *memA, *memB, *memC;
    HANDLE_ERROR(cudaMalloc(&memA, nelems * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&memB, nelems * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&memC, nelems * sizeof(float)));
    float* devA{ static_cast<float*>(memA) };
    float* devB{ static_cast<float*>(memB) };
    float* devC{ static_cast<float*>(memC) };

    // copy A and B to device
    HANDLE_ERROR(cudaMemcpy(devA, hstA, nelems * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(devB, hstB, nelems * sizeof(float), cudaMemcpyHostToDevice));

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {
        start = chr::steady_clock::now();

        globalMatMul<<<grid, block>>>(devA, devB, devC);

        cudaEventRecord(kernelEnd, 0);
        cudaEventSynchronize(kernelEnd);
        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }
    
    // copy back C and check results
    HANDLE_ERROR(cudaMemcpy(hstC, devC, nelems * sizeof(float), cudaMemcpyDeviceToHost));

    // check results and print metrics
    check_results(hstA, hstC, SIZE);
    print_metrics(timings);

    // free resources
    cudaEventDestroy(kernelEnd);

    delete[] hstA;
    delete[] hstB;
    delete[] hstC;

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    
    return 0;
}
