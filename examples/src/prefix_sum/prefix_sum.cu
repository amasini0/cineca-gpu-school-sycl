#include "common_utils.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{    25 };
constexpr size_t SIZE{  2048 }; // <= BLOCK
constexpr size_t BLOCK{ 2048 }; // max 2048

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

__global__ void prescan(const float* in, float* out, float* blockSums, 
                        float* temp, const size_t totalSize)
{
    const size_t block_idx{ blockIdx.x };
    const size_t block_off{ block_idx * BLOCK };
    const size_t thread_idx{ threadIdx.x };

    // load input in shared mem (one thread fills two)
    const size_t idx0{ 2 * thread_idx + block_off };
    const size_t idx1{ idx0 + 1 };
    temp[idx0] = (idx0 < totalSize) ? in[idx0] : 0.0f;
    temp[idx1] = (idx1 < totalSize) ? in[idx1] : 0.0f;

    size_t offset{ 1 }; // start with offset 1, double at each step of upsweep

    // up sweep phase in local mem
    for (size_t d{ BLOCK >> 1 }; d > 0; d >>= 1) {
        // wait for shared mem to be consistent
        __syncthreads();
        // start with SIZE/2 threads, halve at each iteration
        if (thread_idx < d) {
            const size_t ai{ block_off + offset * (2 * thread_idx + 1) - 1 };
            const size_t bi{ block_off + offset * (2 * thread_idx + 2) - 1 };
            temp[bi] += temp[ai];
        }
        // double offset to add to farther elements
        offset *= 2;
    }
    __syncthreads();

    // clear last element
    if (thread_idx == (BLOCK / 2 - 1)) temp[block_off + BLOCK - 1] = 0.0f;

    // traverse down and build block sums
    for (size_t d{1}; d < BLOCK; d *= 2) {
        // start with half offset from before
        offset >>= 1;
        // wait for consistency
        __syncthreads();
        // start with one thread, double at each iteration
        // up to SIZE/2 threads (starting number)
        if (thread_idx < d) {
            const size_t ai{ block_off + offset*(2 * thread_idx + 1) - 1 };
            const size_t bi{ block_off + offset*(2 * thread_idx + 2) - 1 };
            float t{ temp[ai] };
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // write block sums in blockSums array (using single thread)
    if (thread_idx == (BLOCK / 2 - 1)) {
        const size_t last_idx{ BLOCK - 1 + block_off };
        const float last_val{ (last_idx < totalSize) ? in[last_idx] : 0.0f };
        blockSums[blockIdx.x] = temp[block_off + BLOCK - 1] + last_val;
    }

    // write output array
    if (idx0 < totalSize) out[idx0] = temp[idx0];
    if (idx1 < totalSize) out[idx1] = temp[idx1];
}

__global__ void addBlockSum(float* inout, float* blockSums)
{
    const size_t block_idx{ blockIdx.x };
    const size_t block_off{ block_idx * BLOCK };
    const size_t thread_idx{ threadIdx.x };

    // get sum of previous blocks
    float mySum{ blockSums[block_idx] };

    // get indices and check bounds
    const size_t idx0{ 2 * thread_idx + block_off };
    const size_t idx1{ idx0 + 1 };
    if (idx0 < SIZE*SIZE) inout[idx0] += mySum;
    if (idx1 < SIZE*SIZE) inout[idx1] += mySum;
}

///////////////////////////////////////
// main

int main() {

    // arg checks
    if (SIZE > BLOCK) {
        std::cout << "Invalid argument: SIZE must be <= BLOCK" << std::endl;
        return -1;
    }

    constexpr size_t totalSize = SIZE * SIZE;

    // define thread block size and grid
    const auto nblocks{ static_cast<unsigned>(std::ceil(static_cast<float>(totalSize)/BLOCK)) };
    const dim3 grid{ nblocks };  // grid.x = nblocks
    const dim3 block{ BLOCK/2 }; // block.x = BLOCK/2

    // print kernel and launch info
    print_kernel_info("Prefix Sum CUDA", SIZE, 0);
    print_launch_info(grid, block, 0);

    // timers
    chr::time_point<chr::steady_clock> start, stop;
    cudaEvent_t kernelEnd;
    cudaEventCreate(&kernelEnd);
    std::vector<float> timings;
    timings.reserve(RUNS);

    // allocate host arrays
    float* hstIn{  new float[totalSize] };
    float* hstOut{ new float[totalSize] };

    // initialize host arrays
    #pragma omp parallel for
    for (int i = 0; i < totalSize; ++i) {
        hstIn[i]  = 1.0;
    }

    // allocate device arrays
    void *memIn, *memOut, *memSum, *memTmp;
    HANDLE_ERROR(cudaMalloc(&memIn,  totalSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&memOut, totalSize * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&memSum, nblocks * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&memTmp, BLOCK * nblocks * sizeof(float)));
    float* devIn{  static_cast<float*>(memIn)  };
    float* devOut{ static_cast<float*>(memOut) };
    float* preSum{ static_cast<float*>(memSum) };
    float* devTmp{ static_cast<float*>(memTmp) };

    // benchmark kernel + 5 warmup
    for (size_t i{}; i < RUNS; ++i) {

        // copy In to device (reset input)
        HANDLE_ERROR(cudaMemcpy(devIn, hstIn, totalSize * sizeof(float), cudaMemcpyHostToDevice));

        start = chr::steady_clock::now();

        // prescan blocks and compute sum of each
        prescan<<<grid, block>>>(devIn, devOut, preSum, devTmp, totalSize);
        // prescan presums and compute prefix sum
        prescan<<<1, block>>>(preSum, preSum, devIn, devTmp, nblocks);
        // add comulative sums to each block
        addBlockSum<<<grid, block>>>(devOut, preSum);

        cudaEventRecord(kernelEnd, 0);
        cudaEventSynchronize(kernelEnd);
        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy back Out and check results
    HANDLE_ERROR(cudaMemcpy(hstOut, devOut, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

    // check result and print metrics
    check_results(hstOut, SIZE);
    print_metrics(timings);

    // free resources
    cudaEventDestroy(kernelEnd);

    delete[] hstIn;
    delete[] hstOut;

    cudaFree(devIn);
    cudaFree(devOut);
    cudaFree(preSum);

    return 0;
}
