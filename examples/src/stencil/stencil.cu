#include "common_utils.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 }; // use multiple of BLOCK
constexpr size_t BLOCK{  32 }; // use power of 2
constexpr size_t STEPS{ 100 };
constexpr size_t XSOURCE{ SIZE/4 };
constexpr size_t YSOURCE{ SIZE/4 };

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

__global__ void stencilAdd(const float* source, float* target)
{
    const size_t c{ blockIdx.x * blockDim.x + threadIdx.x };
    const size_t r{ blockIdx.y * blockDim.y + threadIdx.y };
    const size_t flat_idx{ r * SIZE + c };
    const float source_val{ source[flat_idx] };
    // leave source unchanged
    if (c == XSOURCE && r == YSOURCE) {
        target[flat_idx] = source_val;
        return;
    }
    // otherwise update heat map
    float temp{ 4.0f * source[flat_idx] };
    temp += (c == 0)        ? source_val : source[flat_idx - 1];
    temp += (c == SIZE - 1) ? source_val : source[flat_idx + 1];
    temp += (r == 0)        ? source_val : source[flat_idx - SIZE];
    temp += (r == SIZE - 1) ? source_val : source[flat_idx + SIZE];
    target[flat_idx] = temp / 8.0f;
}

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE};
    constexpr float min{ 273.0f };
    constexpr float max{ 333.0f };

    // define thread block size and grid
    const auto nblocks{ static_cast<unsigned>(std::ceil(static_cast<float>(SIZE)/BLOCK)) };
    const dim3 grid{ nblocks, nblocks };
    const dim3 block{ BLOCK, BLOCK };

    // print kernel and launch info
    print_kernel_info("2D Stencil Heat Map CUDA", SIZE, 0);
    print_launch_info(grid, block, 0);

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    cudaEvent_t kernelEnd;
    cudaEventCreate(&kernelEnd);
    std::vector<float> timings{};
    timings.reserve(RUNS);

    // allocate host arrays
    float* hstA{ new float[nelems] };
    float* hstB{ new float[nelems] };

    // init host arrays
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            if (i == YSOURCE && j == XSOURCE) {
                hstA[i * SIZE + j] = max;
            } else {
                hstA[i * SIZE + j] = min;
            }
            hstB[i * SIZE + j] = -42.0f;
        }
    }

    // allocate device arrays
    void *memA, *memB;
    HANDLE_ERROR(cudaMalloc(&memA, nelems * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&memB, nelems * sizeof(float)));
    float* devA{ static_cast<float *>(memA) };
    float* devB{ static_cast<float *>(memB) };

    // benchmark kernel runs, 5 for warmup
    for (size_t i{}; i < RUNS; ++i) {

        // copy A and B to device (reset input)
        HANDLE_ERROR(cudaMemcpy(devA, hstA, nelems * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(devB, hstB, nelems * sizeof(float), cudaMemcpyHostToDevice));

        start = chr::steady_clock::now();

        for (size_t s{}; s < STEPS; ++s) {
            float* source{ devA };
            float* target{ devB };
            // swap for odd iterations
            if (s % 2 == 1) {
                target = devA;
                source = devB;
            }

            stencilAdd<<<grid, block>>>(source, target);
        }

        cudaEventRecord(kernelEnd, 0);
        cudaEventSynchronize(kernelEnd);
        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy B back to host and check results
    HANDLE_ERROR(cudaMemcpy(hstB, devB, nelems * sizeof(float), cudaMemcpyDeviceToHost));

    // check result and print metrics
    check_results(hstB, SIZE, min, max);
    print_metrics(timings);

    // free resources
    cudaEventDestroy(kernelEnd);

    delete[] hstA;
    delete[] hstB;

    cudaFree(devA);
    cudaFree(devB);

    return 0;
}
