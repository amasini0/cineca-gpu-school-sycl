#include "common_utils.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{   25 };
constexpr size_t SIZE{ 1024 };
constexpr size_t TILE{   16 };
constexpr size_t VECT{    4 };

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

__global__ void matmulOptGPU(const float* A, const float* B, float* C)
{
    // block index
    const size_t bx{ blockIdx.x };
    const size_t by{ blockIdx.y };

    // thread index
    const size_t tx{ threadIdx.x };
    const size_t ty{ threadIdx.y };

    // tile in shared memory
    __shared__ float As[TILE * TILE];

    // vector to accumulate results
    float cv[TILE] = { 0 };

    const size_t aBegin{ SIZE * TILE * by };
    const size_t aEnd{ aBegin + SIZE - 1 };
    const size_t aStep{  TILE };

    const size_t bBegin{ TILE * VECT * bx };
    const size_t bStep{ TILE * SIZE };

    for (size_t a{ aBegin }, b{ bBegin }; a <= aEnd; a += aStep, b += bStep) {
        for (int i = 0; i < TILE/VECT; ++i) {
            As[(i * VECT + ty) + TILE * tx] = A[a + SIZE * (i * VECT + ty) + tx];
        }
        __syncthreads();

        const float* ap{ &As[0] };
        const float* bp{ &B[b + TILE * ty + tx] };

        for (size_t i{}; i < TILE; ++i) {
            const float bv = bp[0] ;
            for (size_t j{}; j < TILE; ++j) {
                cv[j] += ap[j] * bv;
            }
            ap += TILE;
            bp += SIZE;
        }
        __syncthreads();
    }

    int c = (SIZE * TILE * by) + (TILE * VECT * bx);
    c += TILE * ty + tx;
    for (int i = 0; i < TILE; ++i) {
        C[c] = cv[i];
        c += SIZE;
    }
}

///////////////////////////////////////
// main

int main() {

    constexpr size_t nelems{ SIZE * SIZE };

    // define thread block size and grid
    const dim3 grid{ SIZE / (TILE*VECT), SIZE / TILE };
    const dim3 block{ TILE, VECT };

    // print kernel and launch info
    print_kernel_info("Global Matrix Multiplication CUDA", SIZE, 0);
    print_launch_info(grid, block, 0);

    // timers
    std::chrono::time_point<std::chrono::steady_clock> start{}, stop{};
    cudaEvent_t kernelEnd;
    cudaEventCreate(&kernelEnd);
    std::vector<float> timings;{}
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

        matmulOptGPU<<<grid, block>>>(devA, devB, devC);

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

    free(hstA);
    free(hstB);
    free(hstC);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
