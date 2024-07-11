#include "common_utils.h"
#include "sycl_utils.h"
#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

constexpr size_t RUNS{    25 };
constexpr size_t SIZE{  2048 }; // <= BLOCK
constexpr size_t BLOCK{ 2048 }; // max 2048

namespace chr = std::chrono;

///////////////////////////////////////
// user functions

void prescan(const float* in, float* out, float* blockSums, float* temp, 
             const size_t totalSize, const sycl::nd_item<1> item)
{
    const size_t block_idx{ item.get_group(0) };
    const size_t block_off{ block_idx * BLOCK };
    const size_t thread_idx{ item.get_local_id(0) };

    // load input in shared mem (one thread fills two)
    const size_t idx0{ 2 * thread_idx + block_off };
    const size_t idx1{ idx0 + 1 };
    temp[idx0] = (idx0 < totalSize) ? in[idx0] : 0.0f;
    temp[idx1] = (idx1 < totalSize) ? in[idx1] : 0.0f;

    size_t offset{ 1 }; // start with offset 1, double at each step of upsweep

    // up sweep phase in local mem
    for (size_t d{ BLOCK >> 1 }; d > 0; d >>= 1) {
        // wait for shared mem to be consistent
        item.barrier(sycl::access::fence_space::local_space);
        // start with SIZE/2 threads, halve at each iteration
        if (thread_idx < d) {
            const size_t ai { block_off + offset * (2 * thread_idx + 1) - 1 };
            const size_t bi { block_off + offset * (2 * thread_idx + 2) - 1 };
            temp[bi] += temp[ai];
        }
        // double offset to add to farther elements
        offset *= 2;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // clear last element
    if (thread_idx == (BLOCK / 2 - 1)) temp[block_off + BLOCK - 1] = 0.0f;

    // traverse down and build block sums
    for (size_t d{1}; d < BLOCK; d *= 2) {
        // start with half offset from before
        offset >>= 1;
        // wait for consistency
        item.barrier(sycl::access::fence_space::local_space);
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
    item.barrier(sycl::access::fence_space::local_space);

    // write block sums in blockSums array (using single thread)
    if (thread_idx == (BLOCK / 2 - 1)) {
        const size_t last_idx{ BLOCK - 1 + block_off };
        const float  last_val{ (last_idx < totalSize) ? in[last_idx] : 0.0f };
        blockSums[block_idx] = temp[block_off + BLOCK - 1] + last_val;
    }

    // write output array
    if (idx0 < totalSize) out[idx0] = temp[idx0];
    if (idx1 < totalSize) out[idx1] = temp[idx1];
}

void addBlockSum(float* inout, float* blockSums, const sycl::nd_item<1> item)
{
    const size_t block_idx{ item.get_group(0) };
    const size_t block_off{ block_idx * BLOCK };
    const size_t thread_idx{ item.get_local_id(0) };

    // get sum of previous blocks
    float mySum{ blockSums[block_idx] };
    item.barrier();

    // get indices and check bounds
    const size_t idx0{ 2 * thread_idx + block_off };
    const size_t idx1{ idx0 + 1 };
    if (idx0 < SIZE * SIZE) inout[idx0] += mySum;
    if (idx1 < SIZE * SIZE) inout[idx1] += mySum;
}

///////////////////////////////////////
// main

int main() {

    // arg checks
    if (SIZE > BLOCK) {
        std::cout << "Invalid argument: SIZE must be <= BLOCK" << std::endl;
        return -1;
    }

    constexpr size_t totalSize{ SIZE * SIZE };

    // define work group size and grid
    const size_t nblocks{ static_cast<size_t>(std::ceil(static_cast<float>(totalSize)/BLOCK)) };
    const sycl::range grid{ nblocks };
    const sycl::range block{ BLOCK/2 };

    // init sycl queue with default device
    sycl::queue Q{ sycl::property::queue::in_order() };

    // print kernel and launch info
    print_kernel_info("Prefix Sum SYCL", SIZE, Q);
    print_launch_info(grid, block, Q);   

    // timers
    chr::time_point<chr::steady_clock> start{}, stop{};
    std::vector<float> timings{};
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
    float* devIn{  sycl::malloc_device<float>(totalSize, Q) };
    float* devOut{ sycl::malloc_device<float>(totalSize, Q) };
    float* preSum{ sycl::malloc_device<float>(nblocks, Q) };
    float* devTmp{ sycl::malloc_device<float>(BLOCK * nblocks, Q) };

    // benchmark kernel 100 runs + 5 warmup
    for (size_t i{}; i < RUNS; ++i) {

        // copy In to device (reset input)
        Q.copy<float>(hstIn, devIn, totalSize).wait();

        start = chr::steady_clock::now();

        Q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<1>(grid * block, block),
                [=](sycl::nd_item<1> item) {
                    prescan(devIn, devOut, preSum, devTmp, totalSize, item);
                });
        });

        Q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<1>(block, block),
                [=](sycl::nd_item<1> item) {
                    prescan(preSum, preSum, devIn, devTmp, nblocks, item);
                });
        });

        Q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<1>(grid * block, block),
                [=](sycl::nd_item<1> item) {
                    addBlockSum(devOut, preSum, item);
                });
        });

        Q.wait();
        stop = chr::steady_clock::now();
        timings.push_back(chr::duration<double, std::milli>(stop - start).count());
    }

    // copy back Out and check results
    Q.memcpy(hstOut, devOut, totalSize * sizeof(float)).wait();

    // check result and print metrics
    check_results(hstOut, SIZE);
    print_metrics(timings);    
    
    // free resources
    delete[] hstIn;
    delete[] hstOut;

    sycl::free(devIn, Q);
    sycl::free(devOut, Q);
    sycl::free(preSum, Q);

    return 0;
}
